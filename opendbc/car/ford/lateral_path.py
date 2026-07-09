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
FORD_PATH_DT = 0.05           # LateralMotionControl2 runs at 20Hz


@dataclass(frozen=True)
class LateralPathCommand:
  curvature: float    # c2
  path_angle: float   # c1
  path_offset: float  # c0
  k_meas_filt: float


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
                         steering_pressed: bool, c2_last: float | None = None) -> LateralPathCommand:
  """One 20Hz step of the composed LMC2 command.

  model is a modelV2 reader (or None when missing/stale); k_meas is the
  yaw-derived current curvature, and k_meas_filt is this module's state from
  the previous step. Measured curvature shapes only the transient c0/c1
  residual; c2 remains the upstream lane-following request.
  """
  desired_curvature = _finite(desired_curvature)
  k_meas = _finite(k_meas)
  v_ego = _finite(v_ego)

  if not lat_active:
    return LateralPathCommand(0.0, 0.0, 0.0, k_meas)

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
  if _valid_model_path(model):
    path_angle_raw = _interp(d_look, model.position.x, model.orientation.z)
    path_offset_raw = _interp(d_c0, model.position.x, model.position.y)
  else:
    path_angle_raw = desired_curvature * d_look
    path_offset_raw = 0.5 * desired_curvature * d_c0 * d_c0

  # modelV2's legacy path geometry can be weaker than its lag-adjusted action.
  # While c2 owns lane following, preserve the model residual as-is. Once c2
  # starts fading for a maneuver, require c0/c1 to encode at least the action's
  # constant-curvature arc; keep stronger same-direction model anticipation.
  if c2_share < 1.0:
    path_angle_raw = _authority_floor(path_angle_raw, desired_curvature * d_look)
    path_offset_raw = _authority_floor(path_offset_raw, 0.5 * desired_curvature * d_c0 * d_c0)

  # Subtract delivered curvature only in proportion to the share c2 carries:
  # when c2 is faded out of a maneuver, c1 must hold the arc as an absolute
  # chase heading, or the polynomial reads "straight" mid-turn and unwinds early.
  k_residual = k_meas_filt * _interp(v_ego, FORD_PATH_RESIDUAL_SPEED_BP, (0.0, 1.0)) * c2_share

  c1_error = path_angle_raw / d_look - k_residual
  c1_deadzone = FORD_PATH_C1_DEADZONE + FORD_PATH_C1_CRUISE_DEADZONE * c2_share
  c1_error = math.copysign(max(abs(c1_error) - c1_deadzone, 0.0), c1_error)
  path_angle = _clip(c1_error * d_look, *FORD_PATH_C1_CAN_CLIP)
  # c0 is maneuver-only: on straights it dithers centimeter position commands
  # into the PSCM's offset servo (measured standing left-pull), so it is exactly
  # zero there. Its gate reaches full strength by 0.006 so turn entries get the
  # full offset authority (the 1-share gate withheld 55% through the corridor).
  c0_gain = _interp(abs(desired_curvature), FORD_PATH_C0_GATE_BP, (0.0, 1.0))
  path_offset = _clip((path_offset_raw - 0.5 * k_residual * d_c0 * d_c0) * c0_gain, *FORD_PATH_C0_CAN_CLIP)

  # Don't command a path against the driver's hands: the PSCM integrates the
  # torque fight and discharges it as a jump the moment they release. Zero the
  # whole command so the carcontroller can drop to mode 0 (stock-style relent);
  # the c2 slew then re-engages gently from zero on release.
  if steering_pressed:
    return LateralPathCommand(0.0, 0.0, 0.0, k_meas_filt)

  return LateralPathCommand(c2, path_angle, path_offset, k_meas_filt)
