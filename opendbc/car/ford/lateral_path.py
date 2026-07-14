from __future__ import annotations

from dataclasses import dataclass
import math


# Ford CAN FD composed path controller.
#
# Built on measured F-150 Lightning behavior:
# - c2 (LatCtlCurv_No_Actl) is an absolute curvature request. The PSCM converges
#   measured curvature onto it with a ~1s first-order response and unity DC gain.
#   It does not integrate. (identified from stock mode-1 highway logs)
# - c0/c1 execute fast and carry maneuvers past c2's +/-0.02 1/m range (0.42 rad
#   of path angle delivered a 17m-radius turn), but path following alone showed
#   ~16% proportional droop and no DC disturbance rejection.
# - Yaw-derived measured curvature carries a stable positive bias, so it cannot
#   safely close an integral loop around the absolute c2 lane-following request.
#
# Split by frequency:
# - c2 carries the upstream desired curvature for smooth lane following.
# - c0/c1 carry the transient residual: model path geometry minus the arc the car
#   is already delivering (low-passed measured curvature). They idle at zero in
#   steady state, cover the PSCM's ~1s c2 lag during transients, and hold the
#   remainder as a sustained chase heading past c2's range.

FORD_PATH_C0_CAN_CLIP = (-4.61, 4.60)
FORD_PATH_C1_CAN_CLIP = (-0.475, 0.497)
FORD_PATH_C2_CAN_CLIP = (-0.02, 0.02)

FORD_PATH_D_LOOK_TIME = 1.0   # s, c1 heading sampled this far ahead
FORD_PATH_D_LOOK_MIN = 7.0    # m
FORD_PATH_D_C0 = 7.0          # m, near-field placement lookahead
FORD_PATH_C0_GATE_BP = (0.003, 0.006)  # 1/m of desired curvature, c0 fades in
FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT = 0.02  # 1/m, bounds added offset authority
FORD_PATH_C0_UNDERTRACK_SPEED_BP = (2.0, 5.0, 10.0, 15.0)  # m/s
FORD_PATH_C0_UNDERTRACK_SPEED_GAIN = (0.0, 1.0, 1.0, 0.0)
# A short-horizon filtered copy of persistent maneuver error may add more c0.
# This is deliberately not an integral or global gain: it is capped, gated out
# of lane following, and released quickly when the live error disappears.
FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT = 0.01  # 1/m of additional c0-equivalent curvature
FORD_PATH_C0_ADAPTIVE_ATTACK_TAU = 0.5   # s, requires persistent undertracking
FORD_PATH_C0_ADAPTIVE_RELEASE_TAU = 0.2  # s, remove authority faster than it builds

# Large sustained c2 charges a slow-unwinding state in the PSCM that discharges
# as a pull after the maneuver (observed after every large-c2 turn; absent in
# c2=0 maneuver eras). Confine c2 to cruise/centering duty: full strength on
# gentle curvature, faded to zero in real maneuvers, which c1/c0 carry entirely.
# Band placement bisects the charging threshold: 0.003 was proven safe but
# under-delivered 0.6 m/s^2 in moderate curve tails (lane excursion); 0.02
# sustained provably charges. This tests <=0.006 at full strength.
FORD_PATH_C2_FADE_BP = (0.006, 0.012)  # 1/m of desired curvature
# Stock's wire c2 is heavily rate-limited (p99 step ~1.6e-4/frame); ours carried
# share-transition and engage steps up to 1.3e-3/frame, which read as wheel
# busyness (measured 1.8x stock steering-wheel micro-activity on straights).
FORD_PATH_C2_SLEW = 0.0002    # 1/m per 20Hz frame

FORD_PATH_K_MEAS_TAU = 0.3    # s, low-pass on yaw-derived curvature
FORD_PATH_K_MEAS_MIN_SPEED = 1.0     # m/s, yaw/v curvature is unusable below this
FORD_PATH_RESIDUAL_SPEED_BP = (2.0, 5.0)  # m/s, fade measured-curvature residual in
FORD_PATH_C1_DEADZONE = 0.0003       # 1/m, filtered yaw-noise floor on the c1 curvature error
# At full c2 share, c1's only job is covering real transients: widen its deadzone
# so it does not transmit model plan wiggle the slow c2 channel would filter.
# (measured: straight-road weave amplification 1.85 with c1 active vs 1.30 silent)
FORD_PATH_C1_CRUISE_DEADZONE = 0.0007  # 1/m, added at full c2 share
# C0 and C1 encode the same maneuver path at different polynomial orders. Limit
# same-direction authority buildup by one equivalent-curvature step so model-
# frame jumps remain coherent at any c1 lookahead distance. Release and reversal
# are not delayed. This is a transition bound, not a path gain.
FORD_PATH_MANEUVER_CURVATURE_ATTACK_SLEW = 0.006  # 1/m per 20Hz frame
# A driver hand-back must also bound releases and reversals because the model
# path can differ sharply from the arc the driver was holding. Keep this faster
# than maneuver attack so control resumes promptly without a coefficient step.
FORD_PATH_DRIVER_HANDOFF_CURVATURE_SLEW = 0.01  # 1/m per 20Hz frame
FORD_PATH_DT = 0.05           # LateralMotionControl2 runs at 20Hz


@dataclass(frozen=True)
class LateralPathCommand:
  curvature: float    # c2
  path_angle: float   # c1
  path_offset: float  # c0
  k_meas_filt: float
  c0_undertrack_correction: float
  handoff_complete: bool


def _clip(value: float, lo: float, hi: float) -> float:
  return lo if value < lo else (hi if value > hi else value)


def _interp(x: float, xs, ys) -> float:
  if x <= xs[0]:
    return ys[0]
  if x >= xs[-1]:
    return ys[-1]

  for i in range(1, len(xs)):
    if x < xs[i]:
      t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
  return ys[-1]


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _limit_same_direction_attack(value: float, last: float, max_step: float) -> float:
  """Limit growing same-direction authority without delaying release/reversal."""
  if value * last >= 0.0 and abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _limit_step(value: float, last: float, max_step: float) -> float:
  return _clip(value, last - max_step, last + max_step)


def driver_steering_opposes_command(steering_pressed: bool, steering_torque: float,
                                     desired_curvature: float) -> bool:
  """Select cooperative path tracking when the driver opposes the request.

  Ford's steering-column torque and openpilot curvature use the same turn sign.
  Same-sign torque is the driver helping the path controller and keeps normal
  model control. With no directional request, any pressed input takes priority.
  """
  if not steering_pressed:
    return False

  steering_torque = _finite(steering_torque)
  desired_curvature = _finite(desired_curvature)
  if desired_curvature == 0.0:
    return True
  return steering_torque * desired_curvature < 0.0


def _authority_floor(path_value: float, requested_value: float) -> float:
  """Keep stronger same-direction model geometry, otherwise honor the action."""
  if path_value * requested_value <= 0.0 or abs(path_value) < abs(requested_value):
    return requested_value
  return path_value


def _valid_model_path(model) -> bool:
  if model is None:
    return False
  try:
    return len(model.position.x) > 1 and len(model.position.x) == len(model.position.y) == len(model.orientation.z)
  except (AttributeError, TypeError):
    return False


def lateral_path_command(model, desired_curvature: float, k_meas: float, v_ego: float,
                         k_meas_filt: float, lat_active: bool,
                         driver_override: bool, c2_last: float | None = None,
                         c0_undertrack_correction: float = 0.0,
                         path_angle_last: float | None = None,
                         path_offset_last: float | None = None,
                         driver_handoff: bool = False) -> LateralPathCommand:
  """One 20Hz step of the composed LMC2 command.

  model is a modelV2 reader (or None when missing/stale); k_meas is normally
  yaw-derived curvature and is steering-angle-derived during a driver override.
  k_meas_filt is this module's state from the previous step. Outside cooperative
  control, measured curvature shapes only the transient c0/c1 residual and c2
  remains the upstream lane-following request.
  """
  desired_curvature = _finite(desired_curvature)
  k_meas = _finite(k_meas)
  v_ego = _finite(v_ego)
  c0_undertrack_correction = _clip(_finite(c0_undertrack_correction),
                                   -FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT, FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT)

  if not lat_active:
    return LateralPathCommand(0.0, 0.0, 0.0, k_meas, 0.0, True)

  k_meas_filt = _finite(k_meas_filt)
  if v_ego > FORD_PATH_K_MEAS_MIN_SPEED:
    alpha = 1.0 - math.exp(-FORD_PATH_DT / FORD_PATH_K_MEAS_TAU)
    k_meas_filt = alpha * k_meas + (1.0 - alpha) * k_meas_filt

  # c2: absolute upstream request for smooth lane following. Do not integrate
  # yaw-derived error here: its persistent sensor bias becomes a real pull.
  c2 = _clip(desired_curvature, *FORD_PATH_C2_CAN_CLIP)

  c2_share = _interp(abs(desired_curvature), FORD_PATH_C2_FADE_BP, (1.0, 0.0))
  c2 *= c2_share
  if c2_last is not None:
    c2_last = _finite(c2_last)
    c2 = _clip(c2, c2_last - FORD_PATH_C2_SLEW, c2_last + FORD_PATH_C2_SLEW)

  # c0/c1: model path geometry minus the arc already being delivered. The
  # measured-curvature residual fades out at low speed, where yaw/v is garbage
  # and pure path heading is the proven regime.
  d_look = max(v_ego * FORD_PATH_D_LOOK_TIME, FORD_PATH_D_LOOK_MIN)
  d_c0 = FORD_PATH_D_C0
  if driver_override:
    # Keep the controller synchronized to the arc the driver is physically
    # steering. C0/C1 update with the delivered path while c2 and adaptive
    # authority stay empty, allowing a model hand-back anywhere in the curve
    # without storing a torque fight in the PSCM.
    path_angle = _clip(k_meas * d_look, *FORD_PATH_C1_CAN_CLIP)
    path_offset = _clip(0.5 * k_meas * d_c0 * d_c0, *FORD_PATH_C0_CAN_CLIP)
    return LateralPathCommand(0.0, path_angle, path_offset, k_meas_filt, 0.0, False)

  model_path_valid = _valid_model_path(model)
  if model_path_valid:
    path_angle_raw = _interp(d_look, model.position.x, model.orientation.z)
    path_offset_raw = _interp(d_c0, model.position.x, model.position.y)
  else:
    path_angle_raw = desired_curvature * d_look
    path_offset_raw = 0.5 * desired_curvature * d_c0 * d_c0

  # modelV2's legacy path geometry can be weaker than its lag-adjusted action.
  # While c2 owns lane following, preserve the model residual as-is. Once c2
  # starts fading for a maneuver, c1 follows the bounded action while c0 keeps
  # stronger same-direction near-path geometry and undertracking authority.
  c1_path_angle_raw = path_angle_raw
  if c2_share < 1.0:
    path_angle_raw = _authority_floor(path_angle_raw, desired_curvature * d_look)
    path_offset_raw = _authority_floor(path_offset_raw, 0.5 * desired_curvature * d_c0 * d_c0)
    c1_path_angle_raw = desired_curvature * d_look

  # Subtract only the measured curvature already delivering this path. Bounding
  # it to the path's direction and magnitude makes c0/c1 one-sided: they fill
  # missing authority, but never actively countersteer merely because the truck
  # is still unwinding the previous arc. C2 remains the smooth absolute request.
  path_curvature = path_angle_raw / d_look
  delivered_curvature = k_meas_filt * _interp(v_ego, FORD_PATH_RESIDUAL_SPEED_BP, (0.0, 1.0))
  if delivered_curvature * path_curvature > 0.0:
    delivered_curvature = math.copysign(min(abs(delivered_curvature), abs(path_curvature)), path_curvature)
  else:
    delivered_curvature = 0.0
  # When c2 is faded out of a maneuver, c1 must hold the arc as an absolute
  # chase heading, or the polynomial reads "straight" mid-turn and unwinds early.
  k_residual = delivered_curvature * c2_share

  c1_error = c1_path_angle_raw / d_look - k_residual
  c1_deadzone = FORD_PATH_C1_DEADZONE + FORD_PATH_C1_CRUISE_DEADZONE * c2_share
  c1_error = math.copysign(max(abs(c1_error) - c1_deadzone, 0.0), c1_error)
  path_angle = _clip(c1_error * d_look, *FORD_PATH_C1_CAN_CLIP)
  if c2_share < 1.0 and path_angle_last is not None:
    path_angle_last = _finite(path_angle_last)
    c1_slew = FORD_PATH_MANEUVER_CURVATURE_ATTACK_SLEW * d_look
    path_angle = _limit_same_direction_attack(path_angle, path_angle_last, c1_slew)
  # c0 is maneuver-only: on straights it dithers centimeter position commands
  # into the PSCM's offset servo (measured standing left-pull), so it is exactly
  # zero there. Its gate reaches full strength by 0.006 so turn entries get the
  # full offset authority (the 1-share gate withheld 55% through the corridor).
  c0_gain = _interp(abs(desired_curvature), FORD_PATH_C0_GATE_BP, (0.0, 1.0))
  path_offset = (path_offset_raw - 0.5 * k_residual * d_c0 * d_c0) * c0_gain

  # On logged tight turns, the model/action floor entered path mode correctly
  # but plateaued wide until the driver intervened. Add one bounded c0 arc for
  # the curvature still missing from the truck. This is maneuver-only, fades as
  # measured curvature catches up, and is disabled where yaw/v is unreliable or
  # high-speed c0 is prone to hunting.
  undertrack_error = path_curvature - k_meas_filt
  adaptive_target = 0.0
  if undertrack_error * path_curvature > 0.0:
    undertrack_error = _clip(undertrack_error, -FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT,
                             FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT)
    path_share = 1.0 - c2_share
    speed_gain = _interp(v_ego, FORD_PATH_C0_UNDERTRACK_SPEED_BP, FORD_PATH_C0_UNDERTRACK_SPEED_GAIN)
    gated_error = undertrack_error * path_share * speed_gain
    path_offset += 0.5 * gated_error * d_c0 * d_c0
    if model_path_valid:
      adaptive_target = _clip(gated_error, -FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT,
                              FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT)

  # Bound the state to actual remaining c0 authority so correction cannot wind
  # up invisibly behind the output clip and emerge later during the unwind.
  path_offset = _clip(path_offset, *FORD_PATH_C0_CAN_CLIP)
  adaptive_lo = max(-FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT,
                    2.0 * (FORD_PATH_C0_CAN_CLIP[0] - path_offset) / (d_c0 * d_c0))
  adaptive_hi = min(FORD_PATH_C0_ADAPTIVE_ERROR_LIMIT,
                    2.0 * (FORD_PATH_C0_CAN_CLIP[1] - path_offset) / (d_c0 * d_c0))
  adaptive_target = _clip(adaptive_target, adaptive_lo, adaptive_hi)
  c0_undertrack_correction = _clip(c0_undertrack_correction, adaptive_lo, adaptive_hi)

  # Never carry learned authority into the opposite turn. Skip attacking the
  # new direction for one frame so a reversal starts from the stateless command.
  direction_reversed = c0_undertrack_correction * path_curvature < 0.0
  if direction_reversed:
    c0_undertrack_correction = 0.0
    adaptive_target = 0.0
    applied_correction = 0.0
  else:
    building = adaptive_target * c0_undertrack_correction >= 0.0 and \
               abs(adaptive_target) > abs(c0_undertrack_correction)
    tau = FORD_PATH_C0_ADAPTIVE_ATTACK_TAU if building else FORD_PATH_C0_ADAPTIVE_RELEASE_TAU
    alpha = 1.0 - math.exp(-FORD_PATH_DT / tau)
    applied_correction = c0_undertrack_correction
    c0_undertrack_correction += alpha * (adaptive_target - c0_undertrack_correction)
    if not building:
      applied_correction = c0_undertrack_correction

  path_offset += 0.5 * applied_correction * d_c0 * d_c0

  path_offset = _clip(path_offset, *FORD_PATH_C0_CAN_CLIP)
  if c0_gain > 0.0 and path_offset_last is not None:
    path_offset_last = _finite(path_offset_last)
    c0_slew = 0.5 * FORD_PATH_MANEUVER_CURVATURE_ATTACK_SLEW * d_c0 * d_c0
    path_offset = _limit_same_direction_attack(path_offset, path_offset_last, c0_slew)

  handoff_complete = True
  if driver_handoff and path_angle_last is not None and path_offset_last is not None:
    path_angle_last = _finite(path_angle_last)
    path_offset_last = _finite(path_offset_last)
    target_path_angle = path_angle
    target_path_offset = path_offset
    c1_handoff_slew = FORD_PATH_DRIVER_HANDOFF_CURVATURE_SLEW * d_look
    c0_handoff_slew = 0.5 * FORD_PATH_DRIVER_HANDOFF_CURVATURE_SLEW * d_c0 * d_c0
    path_angle = _limit_step(target_path_angle, path_angle_last, c1_handoff_slew)
    path_offset = _limit_step(target_path_offset, path_offset_last, c0_handoff_slew)
    handoff_complete = path_angle == target_path_angle and path_offset == target_path_offset

  return LateralPathCommand(c2, path_angle, path_offset, k_meas_filt,
                            c0_undertrack_correction, handoff_complete)
