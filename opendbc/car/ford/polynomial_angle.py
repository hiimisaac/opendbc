from __future__ import annotations

from dataclasses import dataclass
import math


# LateralMotionControl2 describes a local path with four coefficients:
#
#   y(s) = c0 + c1*s + 0.5*c2*s^2 + (1/6)*c3*s^3
#
# The PSCM does not expose a wheel-angle command, so this module uses that path
# as one virtual angle actuator. C2 carries a bounded steady road-curvature
# base. C0/C1 are a locked geometric pair carrying the curvature that does not
# fit in C2 plus bounded steering-angle feedback. C3 is the spatial derivative
# of the same virtual curvature state. No coefficient independently fades in or
# out during a maneuver.

FORD_POLY_LOOKAHEAD = 7.0  # m; fixed so C0 and C1 keep one geometric relationship
FORD_POLY_DT = 0.05        # LateralMotionControl2 is sent at 20 Hz

FORD_POLY_C0_CAN_CLIP = (-4.61, 4.60)
FORD_POLY_C1_CAN_CLIP = (-0.475, 0.497)

# Logged large sustained C2 requests charge a slow PSCM state. 0.006/m was the
# largest value reached by the lat-test fade and did not require old-direction
# C2 to re-enter on turn exit. Keep it as a continuously present base instead.
FORD_POLY_C2_BASE_LIMIT = 0.006  # 1/m
FORD_POLY_C2_LATCH_ENTER = 0.012  # 1/m requested curvature
FORD_POLY_C2_LATCH_EXIT_TARGET = 0.002  # 1/m
FORD_POLY_C2_LATCH_EXIT_ERROR = 0.001   # 1/m steering-angle error
FORD_POLY_C2_FLUSH_FRAMES = 10          # 0.5 s at 20 Hz

# C0/C1 extend the virtual target beyond the safe C2 base. Steering-angle
# feedback is bounded separately so it can reject error without becoming an
# unbounded curvature gain.
FORD_POLY_ANGLE_ERROR_LIMIT = 0.01  # 1/m
FORD_POLY_ANGLE_ERROR_DEADZONE_DEG = 0.5
FORD_POLY_ATTACK_SLEW = 0.006       # equivalent 1/m per 20 Hz frame
FORD_POLY_HANDOFF_SLEW = 0.01       # equivalent 1/m per 20 Hz frame

# Curvature-rate requests below this spatial slope are mostly differentiated
# target noise. Releases and reversals bypass the attack filter below.
FORD_POLY_C3_DEADZONE = 0.00005  # 1/m^2
FORD_POLY_C3_ATTACK_TAU = 0.2    # s


@dataclass(frozen=True)
class PolynomialAngleCommand:
  curvature: float       # C2: bounded road-curvature base
  curvature_rate: float  # C3: spatial derivative of virtual_curvature
  path_angle: float      # C1: path_curvature * lookahead
  path_offset: float     # C0: 0.5 * path_curvature * lookahead^2
  virtual_curvature: float
  path_curvature: float
  handoff_complete: bool
  c2_latched: bool
  c2_recovery_frames: int


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _clip(value: float, lo: float, hi: float) -> float:
  return lo if value < lo else (hi if value > hi else value)


def _limit_same_direction_attack(value: float, last: float, max_step: float) -> float:
  """Limit authority buildup while allowing immediate release and reversal."""
  if value * last >= 0.0 and abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _limit_step(value: float, last: float, max_step: float) -> float:
  return _clip(value, last - max_step, last + max_step)


def _path_curvature_clip() -> tuple[float, float]:
  """Intersect C0/C1 wire limits so the locked pair never distorts."""
  c0_lo = 2.0 * FORD_POLY_C0_CAN_CLIP[0] / (FORD_POLY_LOOKAHEAD ** 2)
  c0_hi = 2.0 * FORD_POLY_C0_CAN_CLIP[1] / (FORD_POLY_LOOKAHEAD ** 2)
  c1_lo = FORD_POLY_C1_CAN_CLIP[0] / FORD_POLY_LOOKAHEAD
  c1_hi = FORD_POLY_C1_CAN_CLIP[1] / FORD_POLY_LOOKAHEAD
  return max(c0_lo, c1_lo), min(c0_hi, c1_hi)


FORD_POLY_PATH_CURVATURE_CLIP = _path_curvature_clip()


def polynomial_angle_command(desired_curvature: float, angle_error_curvature: float,
                             measured_curvature: float, v_ego: float,
                             lat_active: bool, driver_override: bool,
                             *, virtual_curvature_last: float = 0.0,
                             path_curvature_last: float = 0.0,
                             curvature_rate_last: float = 0.0,
                             driver_handoff: bool = False,
                             c2_latched_last: bool = False,
                             c2_recovery_frames_last: int = 0) -> PolynomialAngleCommand:
  """Encode one virtual steering target across the complete Ford polynomial.

  desired_curvature is openpilot's lag-adjusted action. angle_error_curvature
  is the desired-minus-measured steering-wheel angle expressed as curvature.
  During an override, measured_curvature becomes the target so the transmitted
  path follows the arc held by the driver instead of fighting it.
  """
  desired_curvature = _finite(desired_curvature)
  angle_error_curvature = _finite(angle_error_curvature)
  measured_curvature = _finite(measured_curvature)
  v_ego = max(_finite(v_ego), 1.0)
  virtual_curvature_last = _finite(virtual_curvature_last)
  path_curvature_last = _finite(path_curvature_last)
  curvature_rate_last = _finite(curvature_rate_last)
  c2_latched = bool(c2_latched_last)
  c2_recovery_frames = max(int(c2_recovery_frames_last), 0)

  if not lat_active:
    # Continue sending zero while preserving the flush latch across a quick
    # disengage/re-engage. After half a second of inactive zero commands, C2 is
    # allowed to serve gentle curvature again.
    if c2_latched:
      c2_recovery_frames += 1
      if c2_recovery_frames >= FORD_POLY_C2_FLUSH_FRAMES:
        c2_latched = False
        c2_recovery_frames = 0
    return PolynomialAngleCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True,
                                  c2_latched, c2_recovery_frames)

  target_curvature = measured_curvature if driver_override else desired_curvature
  if driver_override:
    # Synchronize directly with the wheel the driver is holding.
    virtual_curvature = target_curvature
  elif driver_handoff:
    # Bound both directions while crossing from the held arc to model control.
    virtual_curvature = _limit_step(target_curvature, virtual_curvature_last,
                                    FORD_POLY_HANDOFF_SLEW)
  else:
    virtual_curvature = _limit_same_direction_attack(target_curvature, virtual_curvature_last,
                                                     FORD_POLY_ATTACK_SLEW)

  # Large-turn C2 is stateful by observation, so remove it for the complete
  # maneuver rather than fading it according to instantaneous curvature. It
  # may return only after target and steering-angle error have both remained
  # near straight long enough to flush the PSCM's held-curvature state.
  if abs(target_curvature) >= FORD_POLY_C2_LATCH_ENTER:
    c2_latched = True
    c2_recovery_frames = 0
  elif c2_latched:
    near_straight = abs(target_curvature) <= FORD_POLY_C2_LATCH_EXIT_TARGET and \
                    abs(angle_error_curvature) <= FORD_POLY_C2_LATCH_EXIT_ERROR
    if near_straight:
      c2_recovery_frames += 1
      if c2_recovery_frames >= FORD_POLY_C2_FLUSH_FRAMES:
        c2_latched = False
        c2_recovery_frames = 0
    else:
      c2_recovery_frames = 0

  # Outside maneuvers C2 is a continuous saturated projection of the shared
  # target. While latched, its entire allocation moves into the C0/C1 pair.
  c2 = 0.0 if c2_latched else _clip(virtual_curvature,
                                    -FORD_POLY_C2_BASE_LIMIT, FORD_POLY_C2_BASE_LIMIT)
  overflow_curvature = virtual_curvature - c2

  angle_correction = 0.0 if driver_override else _clip(angle_error_curvature,
                                                       -FORD_POLY_ANGLE_ERROR_LIMIT,
                                                       FORD_POLY_ANGLE_ERROR_LIMIT)
  path_target = _clip(overflow_curvature + angle_correction,
                      *FORD_POLY_PATH_CURVATURE_CLIP)

  if driver_override:
    path_curvature = path_target
  elif driver_handoff:
    path_curvature = _limit_step(path_target, path_curvature_last,
                                 FORD_POLY_HANDOFF_SLEW)
  else:
    path_curvature = _limit_same_direction_attack(path_target, path_curvature_last,
                                                  FORD_POLY_ATTACK_SLEW)
  path_curvature = _clip(path_curvature, *FORD_POLY_PATH_CURVATURE_CLIP)
  handoff_complete = not driver_handoff or \
                     (virtual_curvature == target_curvature and path_curvature == path_target)

  c1 = path_curvature * FORD_POLY_LOOKAHEAD
  c0 = 0.5 * path_curvature * FORD_POLY_LOOKAHEAD ** 2

  # A temporal change in the moving target becomes a spatial curvature slope.
  # Unlike the old model-path estimate, this coefficient cannot disagree with
  # the target direction or retain the previous maneuver after target release.
  if driver_override:
    c3 = 0.0
  else:
    raw_c3 = (virtual_curvature - virtual_curvature_last) / (v_ego * FORD_POLY_DT)
    if abs(raw_c3) < FORD_POLY_C3_DEADZONE:
      raw_c3 = 0.0

    # Smooth only increasing same-direction slope. A release or reversal must
    # reach the PSCM immediately to avoid recreating the old unwind defect.
    if raw_c3 * curvature_rate_last >= 0.0 and abs(raw_c3) > abs(curvature_rate_last):
      alpha = 1.0 - math.exp(-FORD_POLY_DT / FORD_POLY_C3_ATTACK_TAU)
      c3 = curvature_rate_last + alpha * (raw_c3 - curvature_rate_last)
    else:
      c3 = raw_c3

  return PolynomialAngleCommand(c2, c3, c1, c0, virtual_curvature,
                                path_curvature, handoff_complete,
                                c2_latched, c2_recovery_frames)
