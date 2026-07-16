from __future__ import annotations

from collections import deque
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
# - c3 carries the model path's spatial curvature slope over the near field.
# - c0/c1 carry the transient residual: model path geometry minus the arc the car
#   is already delivering (low-passed measured curvature). They idle at zero in
#   steady state, cover the PSCM's ~1s c2 lag during transients, and hold the
#   remainder as a sustained chase heading past c2's range.
# - c0 also closes bounded steering-angle error during maneuvers. Actual wheel
#   motion is projected 0.1 s forward first, so feedback does not keep building
#   while the PSCM is already moving toward the requested angle.

FORD_PATH_C0_CAN_CLIP = (-4.61, 4.60)
FORD_PATH_C1_CAN_CLIP = (-0.475, 0.497)
FORD_PATH_C2_CAN_CLIP = (-0.02, 0.02)

FORD_PATH_D_LOOK_TIME = 1.0   # s, c1 heading sampled this far ahead
FORD_PATH_D_LOOK_MIN = 7.0    # m
FORD_PATH_D_C0 = 7.0          # m, near-field placement lookahead
FORD_PATH_C0_GATE_BP = (0.003, 0.006)  # 1/m of desired curvature, c0 fades in
FORD_PATH_C0_FEEDBACK_ERROR_LIMIT = 0.02  # 1/m of steering-angle tracking error

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
# Once a real maneuver has removed c2, keep it out until both the action and
# physical wheel have remained near straight. This matches the PSCM's observed
# statefulness instead of letting the instantaneous fade recharge old-turn c2
# on the way out of a curve.
FORD_PATH_C2_LATCH_ENTER = FORD_PATH_C2_FADE_BP[1]
FORD_PATH_C2_LATCH_EXIT_TARGET = 0.002  # 1/m
FORD_PATH_C2_LATCH_EXIT_ERROR = 0.001   # 1/m of steering-angle error
FORD_PATH_C2_FLUSH_FRAMES = 10          # 0.5 s at 20Hz

# Only while the large-turn latch is flushing, steering-angle error may ask
# c0/c1 to return the wheel toward its upstream target. It is separately capped
# and attack-limited; release and reversal remain immediate.
FORD_PATH_UNWIND_ERROR_LIMIT = 0.006   # 1/m
FORD_PATH_UNWIND_ATTACK_SLEW = 0.002   # 1/m per 20Hz frame
FORD_PATH_UNWIND_ANGLE_DEADZONE_DEG = 2.5

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
# C3 behaves like a direct steering-rate request in the PSCM. Do not let one
# noisy model frame step straight to the wire limit, but preserve the complete
# sustained spatial slope and never delay a release or reversal.
FORD_PATH_C3_ATTACK_SLEW = 0.0002  # 1/m^2 per 20Hz frame
# C3 comes from one model frame at three near-field distances. Same-direction
# estimates keep the full median slope; conflicting horizons continuously
# attenuate it without timers, latches, or a new authority cap.
FORD_PATH_C3_HORIZONS = (3.5, 5.0, 7.0)  # m
# A driver hand-back must also bound releases and reversals because the model
# path can differ sharply from the arc the driver was holding. Keep this faster
# than maneuver attack so control resumes promptly without a coefficient step.
FORD_PATH_DRIVER_HANDOFF_CURVATURE_SLEW = 0.01  # 1/m per 20Hz frame
FORD_PATH_DT = 0.05           # LateralMotionControl2 runs at 20Hz
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.1  # s, measured wheel-motion lookahead


@dataclass(frozen=True)
class LateralPathCommand:
  curvature: float    # c2
  curvature_rate: float  # c3, spatial derivative of curvature
  path_angle: float   # c1
  path_offset: float  # c0
  k_meas_filt: float
  handoff_complete: bool
  c2_latched: bool = False
  c2_recovery_frames: int = 0
  unwind_curvature: float = 0.0


class SteeringAngleProjector:
  """Project steering angle from a short, fixed-rate measurement window."""

  def __init__(self, sample_dt: float = FORD_PATH_DT,
               horizon: float = FORD_PATH_ANGLE_PROJECTION_HORIZON):
    self.sample_dt = max(_finite(sample_dt), FORD_PATH_DT)
    self.horizon = max(_finite(horizon), 0.0)
    sample_count = max(round(self.horizon / self.sample_dt) + 1, 2)
    self.samples: deque[float] = deque(maxlen=sample_count)

  def update(self, actual_angle_deg: float) -> float:
    fallback = self.samples[-1] if self.samples else 0.0
    actual_angle_deg = _finite(actual_angle_deg, fallback)
    self.samples.append(actual_angle_deg)
    sample_time = (len(self.samples) - 1) * self.sample_dt
    if sample_time <= 0.0:
      return actual_angle_deg

    steering_rate_deg_s = (self.samples[-1] - self.samples[0]) / sample_time
    return actual_angle_deg + steering_rate_deg_s * self.horizon


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


def projected_tracking_error(target: float, current: float, projected: float) -> float:
  """Discount tracking error only when measured motion is closing the target."""
  error = _finite(target) - _finite(current)
  projected_error = _finite(target) - _finite(projected)
  projected_motion = _finite(projected) - _finite(current)
  if error == 0.0 or projected_motion * error <= 0.0:
    return error
  if projected_error * error <= 0.0:
    return 0.0
  return projected_error if abs(projected_error) < abs(error) else error


def driver_steering_opposes_command(steering_pressed: bool, steering_torque: float,
                                     steering_angle_error_deg: float) -> bool:
  """Select cooperative path tracking when the driver opposes the request.

  Compare two signals in steering-wheel coordinates. Ford curvature has the
  opposite sign from steering angle, which made the old curvature comparison
  classify a driver helping the requested wheel motion as an override. With no
  requested wheel motion, any pressed input takes priority.
  """
  if not steering_pressed:
    return False

  steering_torque = _finite(steering_torque)
  steering_angle_error_deg = _finite(steering_angle_error_deg)
  if steering_angle_error_deg == 0.0:
    return True
  return steering_torque * steering_angle_error_deg < 0.0


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


def model_curvature_rate(model, horizon: float) -> float:
  """Estimate d(curvature)/d(distance) from model path heading.

  For a constant spatial curvature slope q, path heading follows
  psi(s) = psi0 + k0*s + 0.5*q*s^2. Sampling the heading at the start,
  midpoint, and end therefore recovers q while canceling heading and
  curvature offsets. Distance is accumulated along the model path rather than
  using longitudinal x so the result remains a spatial derivative in turns.
  """
  if not _valid_model_path(model):
    return 0.0

  try:
    xs = [float(x) for x in model.position.x]
    ys = [float(y) for y in model.position.y]
    headings = [float(heading) for heading in model.orientation.z]
  except (AttributeError, TypeError, ValueError):
    return 0.0

  if not all(math.isfinite(value) for values in (xs, ys, headings) for value in values):
    return 0.0

  distances = [0.0]
  for i in range(1, len(xs)):
    distances.append(distances[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))

  horizon = _finite(horizon)
  horizon = min(horizon, distances[-1])
  if horizon <= 0.0:
    return 0.0

  # Unwrap before interpolation so a +/-pi crossing cannot look like an
  # enormous reversal in spatial curvature slope.
  unwrapped_headings = [headings[0]]
  for heading in headings[1:]:
    delta = (heading - unwrapped_headings[-1] + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped_headings.append(unwrapped_headings[-1] + delta)

  heading_start = unwrapped_headings[0]
  heading_mid = _interp(0.5 * horizon, distances, unwrapped_headings)
  heading_end = _interp(horizon, distances, unwrapped_headings)
  return _finite(4.0 * (heading_start - 2.0 * heading_mid + heading_end) / (horizon * horizon))


def model_curvature_rate_consensus(model) -> float:
  """Return the median near-field spatial slope, attenuated by sign conflict."""
  curvature_rates = tuple(model_curvature_rate(model, horizon) for horizon in FORD_PATH_C3_HORIZONS)
  magnitude = sum(abs(curvature_rate) for curvature_rate in curvature_rates)
  if magnitude == 0.0:
    return 0.0

  median = sorted(curvature_rates)[1]
  agreement = abs(sum(curvature_rates)) / magnitude
  return _finite(median * agreement)


def lateral_path_command(model, desired_curvature: float, k_meas: float, v_ego: float,
                         k_meas_filt: float, lat_active: bool,
                         driver_override: bool, c2_last: float | None = None,
                         path_angle_last: float | None = None,
                         path_offset_last: float | None = None,
                         driver_handoff: bool = False,
                         angle_error_curvature: float = 0.0,
                         wheel_curvature: float = 0.0,
                         projected_wheel_curvature: float | None = None,
                         c2_latched_last: bool = False,
                         c2_recovery_frames_last: int = 0,
                         unwind_curvature_last: float = 0.0,
                         curvature_rate_last: float | None = None) -> LateralPathCommand:
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
  angle_error_curvature = _finite(angle_error_curvature)
  wheel_curvature = _finite(wheel_curvature)
  steering_feedback_available = projected_wheel_curvature is not None
  if projected_wheel_curvature is None:
    projected_wheel_curvature = wheel_curvature
  projected_wheel_curvature = _finite(projected_wheel_curvature, wheel_curvature)
  c2_latched = bool(c2_latched_last)
  c2_recovery_frames = max(int(c2_recovery_frames_last), 0)
  unwind_curvature_last = _clip(_finite(unwind_curvature_last),
                                -FORD_PATH_UNWIND_ERROR_LIMIT, FORD_PATH_UNWIND_ERROR_LIMIT)

  if not lat_active:
    # Preserve the c2 flush across a quick disengage/re-engage. Ten inactive
    # zero frames are also sufficient to clear the observed held state.
    if c2_latched:
      c2_recovery_frames += 1
      if c2_recovery_frames >= FORD_PATH_C2_FLUSH_FRAMES:
        c2_latched = False
        c2_recovery_frames = 0
    return LateralPathCommand(0.0, 0.0, 0.0, 0.0, k_meas, True,
                              c2_latched, c2_recovery_frames, 0.0)

  k_meas_filt = _finite(k_meas_filt)
  if v_ego > FORD_PATH_K_MEAS_MIN_SPEED:
    alpha = 1.0 - math.exp(-FORD_PATH_DT / FORD_PATH_K_MEAS_TAU)
    k_meas_filt = alpha * k_meas + (1.0 - alpha) * k_meas_filt

  # Large-turn c2 is stateful by observation. Once removed, do not let the
  # falling target fade it back into the old turn; require a stable target and
  # physical steering-angle flush before returning c2 to cruise duty.
  if abs(desired_curvature) >= FORD_PATH_C2_LATCH_ENTER:
    c2_latched = True
    c2_recovery_frames = 0
  elif c2_latched:
    near_straight = not driver_override and \
                    abs(desired_curvature) <= FORD_PATH_C2_LATCH_EXIT_TARGET and \
                    abs(angle_error_curvature) <= FORD_PATH_C2_LATCH_EXIT_ERROR
    if near_straight:
      c2_recovery_frames += 1
      if c2_recovery_frames >= FORD_PATH_C2_FLUSH_FRAMES:
        c2_latched = False
        c2_recovery_frames = 0
    else:
      c2_recovery_frames = 0

  # c2: absolute upstream request for smooth lane following. Do not integrate
  # yaw-derived error here: its persistent sensor bias becomes a real pull.
  instantaneous_c2_share = _interp(abs(desired_curvature), FORD_PATH_C2_FADE_BP, (1.0, 0.0))
  c2_share = 0.0 if c2_latched else instantaneous_c2_share
  c2 = 0.0 if c2_latched else _clip(desired_curvature, *FORD_PATH_C2_CAN_CLIP) * c2_share
  if not c2_latched and c2_last is not None:
    c2_last = _finite(c2_last)
    # The PSCM retains large c2 turns. Limit only same-direction authority
    # buildup; release immediately, and flush the old sign before reversal.
    if c2 * c2_last < 0.0:
      c2 = 0.0
    else:
      c2 = _limit_same_direction_attack(c2, c2_last, FORD_PATH_C2_SLEW)

  # c0/c1: model path geometry minus the arc already being delivered. The
  # measured-curvature residual fades out at low speed, where yaw/v is garbage
  # and pure path heading is the proven regime.
  d_look = max(v_ego * FORD_PATH_D_LOOK_TIME, FORD_PATH_D_LOOK_MIN)
  d_c0 = FORD_PATH_D_C0
  if driver_override:
    # Keep the controller synchronized to the arc the driver is physically
    # steering. C0/C1 update with the delivered path while c2 and feedback
    # authority stay empty, allowing a model hand-back anywhere in the curve
    # without storing a torque fight in the PSCM.
    path_angle = _clip(k_meas * d_look, *FORD_PATH_C1_CAN_CLIP)
    path_offset = _clip(0.5 * k_meas * d_c0 * d_c0, *FORD_PATH_C0_CAN_CLIP)
    return LateralPathCommand(0.0, 0.0, path_angle, path_offset, k_meas_filt, False,
                              c2_latched, c2_recovery_frames, 0.0)

  model_path_valid = _valid_model_path(model)
  curvature_rate = model_curvature_rate_consensus(model) if model_path_valid else 0.0
  if model_path_valid:
    path_angle_raw = _interp(d_look, model.position.x, model.orientation.z)
    path_offset_raw = _interp(d_c0, model.position.x, model.position.y)
  else:
    path_angle_raw = desired_curvature * d_look
    path_offset_raw = 0.5 * desired_curvature * d_c0 * d_c0

  # A moving turn may already be visible in the spatial polynomial while the
  # live action is still gentle. Preview only the shortest c3 horizon, and only
  # when c0 and c1 independently confirm a building maneuver in the same
  # direction. This starts a real curvature reversal sooner without admitting
  # isolated c3 plan noise into ordinary lane following.
  coherent_preview_curvature = 0.0
  if model_path_valid:
    preview_curvature = desired_curvature + curvature_rate * FORD_PATH_C3_HORIZONS[0]
    c0_path_curvature = 2.0 * path_offset_raw / (d_c0 * d_c0)
    c1_path_curvature = path_angle_raw / d_look
    preview_direction_agrees = preview_curvature * curvature_rate > 0.0 and \
                               abs(preview_curvature) > abs(desired_curvature)
    geometry_direction_agrees = c0_path_curvature * preview_curvature > 0.0 and \
                                 c1_path_curvature * preview_curvature > 0.0
    geometry_is_maneuver = abs(c0_path_curvature) >= FORD_PATH_C0_GATE_BP[0] and \
                           abs(c1_path_curvature) >= FORD_PATH_C0_GATE_BP[0]
    moving_reversal = v_ego > FORD_PATH_K_MEAS_MIN_SPEED and k_meas * preview_curvature < 0.0
    if moving_reversal and preview_direction_agrees and geometry_direction_agrees and geometry_is_maneuver:
      coherent_preview_curvature = preview_curvature

  # Small c3 zero-crossings act like direct steering-rate commands in the PSCM
  # and made gentle lane following chatty. Use the existing c2 maneuver blend,
  # augmented by coherent imminent geometry. Do not couple this blend to the
  # historical c2 latch: doing so forced full c3 into gentle exits.
  live_maneuver_share = 1.0 - instantaneous_c2_share
  preview_maneuver_share = _interp(abs(coherent_preview_curvature), FORD_PATH_C2_FADE_BP, (0.0, 1.0))
  curvature_rate *= max(live_maneuver_share, preview_maneuver_share)

  if curvature_rate_last is not None:
    curvature_rate_last = _finite(curvature_rate_last)
    curvature_rate = _limit_same_direction_attack(curvature_rate, curvature_rate_last,
                                                   FORD_PATH_C3_ATTACK_SLEW)

  # modelV2's legacy path geometry can be weaker than its lag-adjusted action.
  # While c2 owns lane following, preserve the model residual as-is. Once c2
  # starts fading for a maneuver, c1 follows the bounded action while c0 keeps
  # stronger same-direction near-path geometry and steering-angle feedback.
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
  # zero unless the live action or coherent imminent geometry is a maneuver.
  # Its gate reaches full strength by 0.006 so turn entries get the full offset
  # authority (the 1-share gate withheld 55% through the corridor).
  c0_gate_curvature = max(abs(desired_curvature), abs(coherent_preview_curvature))
  c0_gain = _interp(c0_gate_curvature, FORD_PATH_C0_GATE_BP, (0.0, 1.0))
  path_offset = (path_offset_raw - 0.5 * k_residual * d_c0 * d_c0) * c0_gain

  # The normal residual is intentionally one-sided so biased yaw cannot create
  # countersteer. During a latched large-turn exit only, the steering wheel is
  # a trustworthy error sensor: permit a bounded correction when it points
  # toward the desired angle and opposite the wheel's remaining old-turn arc.
  unwind_target = 0.0
  if c2_latched and angle_error_curvature * wheel_curvature < 0.0:
    unwind_target = _clip(angle_error_curvature,
                          -FORD_PATH_UNWIND_ERROR_LIMIT, FORD_PATH_UNWIND_ERROR_LIMIT)
  unwind_curvature = _limit_same_direction_attack(unwind_target, unwind_curvature_last,
                                                   FORD_PATH_UNWIND_ATTACK_SLEW)

  # Steering-angle feedback closes the full near-field model target, including
  # c0 geometry stronger than action curvature. Projected wheel motion can only
  # reduce the instantaneous error; it can never amplify or reverse it.
  feedback_target_curvature = 2.0 * path_offset_raw / (d_c0 * d_c0)
  feedback_error = projected_tracking_error(feedback_target_curvature, wheel_curvature,
                                            projected_wheel_curvature)
  feedback_error = _clip(feedback_error, -FORD_PATH_C0_FEEDBACK_ERROR_LIMIT,
                         FORD_PATH_C0_FEEDBACK_ERROR_LIMIT)
  if steering_feedback_available and unwind_target == 0.0 and feedback_error * feedback_target_curvature > 0.0:
    path_offset += 0.5 * feedback_error * (1.0 - c2_share) * d_c0 * d_c0

  path_angle = _clip(path_angle + unwind_curvature * d_look, *FORD_PATH_C1_CAN_CLIP)
  path_offset += 0.5 * unwind_curvature * d_c0 * d_c0
  # Keep c3 when it assists the physical unwind, but suppress a model slope
  # that would push back into the wheel's remaining old-turn direction.
  if unwind_curvature != 0.0 and curvature_rate * unwind_curvature < 0.0:
    curvature_rate = 0.0

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

  return LateralPathCommand(c2, curvature_rate, path_angle, path_offset, k_meas_filt,
                            handoff_complete,
                            c2_latched, c2_recovery_frames, unwind_curvature)
