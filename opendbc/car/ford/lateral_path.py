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
# - Yaw-derived measured curvature carries a stable ~+0.0003 1/m bias.
#
# Split by frequency:
# - c2 carries everything sustained: desired curvature plus a slow integral trim
#   that nulls sensor bias, road crown, and PSCM droop.
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

FORD_PATH_K_MEAS_TAU = 0.3    # s, low-pass on yaw-derived curvature
FORD_PATH_K_MEAS_MIN_SPEED = 1.0     # m/s, yaw/v curvature is unusable below this
FORD_PATH_RESIDUAL_SPEED_BP = (2.0, 5.0)  # m/s, fade measured-curvature residual in
FORD_PATH_C1_DEADZONE = 0.0005       # 1/m, yaw-noise floor on the c1 curvature error
# Trim is a bias estimator, not a transient chaser: it learns only near-straight
# with a tens-of-seconds time constant. A hot trim charges on turn-entry lag and
# discharges as a pull on the next straight (observed on the first drive).
FORD_PATH_TRIM_KI = 0.05      # 1/s, integral trim rate toward (desired - measured)
FORD_PATH_TRIM_CLIP = 0.0015  # 1/m, measured DC bias is ~0.0003-0.001
FORD_PATH_TRIM_ERR_CLIP = 0.002      # 1/m, bounds the learning step
FORD_PATH_TRIM_MAX_CURVATURE = 0.0015  # 1/m, learn on straights only
FORD_PATH_TRIM_MIN_SPEED = 5.0       # m/s
FORD_PATH_DT = 0.05           # LateralMotionControl2 runs at 20Hz


@dataclass(frozen=True)
class LateralPathCommand:
  curvature: float    # c2
  path_angle: float   # c1
  path_offset: float  # c0
  k_meas_filt: float
  trim: float


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


def _valid_model_path(model) -> bool:
  if model is None:
    return False
  try:
    return len(model.position.x) > 1 and len(model.position.x) == len(model.position.y) == len(model.orientation.z)
  except (AttributeError, TypeError):
    return False


def lateral_path_command(model, desired_curvature: float, k_meas: float, v_ego: float,
                         k_meas_filt: float, trim: float, lat_active: bool,
                         steering_pressed: bool) -> LateralPathCommand:
  """One 20Hz step of the composed LMC2 command.

  model is a modelV2 reader (or None when missing/stale); k_meas is the
  yaw-derived current curvature; k_meas_filt and trim are this module's state
  from the previous step. The trim persists across engagements: it estimates
  slow disturbances (sensor bias, crown), not maneuver state.
  """
  desired_curvature = _finite(desired_curvature)
  k_meas = _finite(k_meas)
  v_ego = _finite(v_ego)
  trim = _clip(_finite(trim), -FORD_PATH_TRIM_CLIP, FORD_PATH_TRIM_CLIP)

  if not lat_active:
    return LateralPathCommand(0.0, 0.0, 0.0, k_meas, trim)

  k_meas_filt = _finite(k_meas_filt)
  if v_ego > FORD_PATH_K_MEAS_MIN_SPEED:
    alpha = 1.0 - math.exp(-FORD_PATH_DT / FORD_PATH_K_MEAS_TAU)
    k_meas_filt = alpha * k_meas + (1.0 - alpha) * k_meas_filt

  # c2: absolute request, slow integral trim for DC disturbances
  c2_raw = desired_curvature + trim
  c2 = _clip(c2_raw, *FORD_PATH_C2_CAN_CLIP)
  c2_railed = c2 != c2_raw
  if v_ego > FORD_PATH_TRIM_MIN_SPEED and not steering_pressed and not c2_railed and \
     abs(desired_curvature) < FORD_PATH_TRIM_MAX_CURVATURE:
    trim_err = _clip(desired_curvature - k_meas_filt, -FORD_PATH_TRIM_ERR_CLIP, FORD_PATH_TRIM_ERR_CLIP)
    trim = _clip(trim + FORD_PATH_TRIM_KI * trim_err * FORD_PATH_DT,
                 -FORD_PATH_TRIM_CLIP, FORD_PATH_TRIM_CLIP)

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

  k_residual = k_meas_filt * _interp(v_ego, FORD_PATH_RESIDUAL_SPEED_BP, (0.0, 1.0))

  c1_error = path_angle_raw / d_look - k_residual
  c1_error = math.copysign(max(abs(c1_error) - FORD_PATH_C1_DEADZONE, 0.0), c1_error)
  path_angle = _clip(c1_error * d_look, *FORD_PATH_C1_CAN_CLIP)
  path_offset = _clip(path_offset_raw - 0.5 * k_residual * d_c0 * d_c0, *FORD_PATH_C0_CAN_CLIP)

  # Don't command a path against the driver's hands: the PSCM integrates the
  # torque fight and discharges it as a pull for seconds after release.
  if steering_pressed:
    path_angle = 0.0
    path_offset = 0.0

  return LateralPathCommand(c2, path_angle, path_offset, k_meas_filt, trim)
