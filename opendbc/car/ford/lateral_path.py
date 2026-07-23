"""Ford LMC2 path controller."""

from collections import deque
from dataclasses import dataclass
import math


PATH_C0_LIMITS = (-4.61, 4.60)
PATH_C1_LIMITS = (-0.475, 0.497)
PATH_C2_LIMITS = (-0.02, 0.02)
PATH_C3_LIMITS = (-0.001024, 0.001023)

PATH_MIN_LOOKAHEAD = 7.0

PATH_C2_FADE_BP = (0.006, 0.012)
PATH_C2_SETTLED_BP = (0.003, 0.006)
PATH_C2_SLEW = 0.0002
PATH_MODEL_MANEUVER_MIN = 0.003
PATH_MODEL_GEOMETRY_FULL = 0.012
PATH_TRACKING_ERROR_DEADZONE = 0.0005
PATH_C0_TRACKING_ERROR_LIMIT = 0.02
PATH_C1_TRACKING_ERROR_LIMIT = 0.012
PATH_PREVIEW_CONFLICT_BP = (0.006, 0.012)
PATH_PREVIEW_ACTION_RELEASE_BP = (0.0015, 0.003)
PATH_PREVIEW_RELEASE_COMMAND_BP = (0.001, 0.003)
PATH_PREVIEW_CONFLICT_SPEED_BP = (2.0, 3.0)
PATH_C3_UNWIND_ERROR_BP = (0.0005, 0.002)
PATH_C3_UNWIND_TARGET_MIN = 0.003
PATH_UNWIND_ERROR_DEADZONE = 0.0005
PATH_UNWIND_LIMIT = 0.006
PATH_MANEUVER_CURVATURE_SLEW = 0.006
PATH_C3_SLEW = 0.0002
FORD_PATH_DT = 0.05
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.35


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


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
  """Select cooperative path tracking when the driver opposes the request."""
  if not steering_pressed:
    return False

  steering_torque = _finite(steering_torque)
  steering_angle_error_deg = _finite(steering_angle_error_deg)
  if steering_angle_error_deg == 0.0:
    return True
  return steering_torque * steering_angle_error_deg < 0.0


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


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _interp(value: float, lower: float, upper: float, lower_value: float, upper_value: float) -> float:
  if value <= lower:
    return lower_value
  if value >= upper:
    return upper_value
  alpha = (value - lower) / (upper - lower)
  return lower_value + alpha * (upper_value - lower_value)


def _deadzone(value: float, deadzone: float) -> float:
  return math.copysign(max(abs(value) - deadzone, 0.0), value)


def _blend(first: float, second: float, second_share: float) -> float:
  return first + _clip(second_share, (0.0, 1.0)) * (second - first)


def _limit_attack(value: float, last: float, max_step: float) -> float:
  """Bound authority growth, release immediately, and reverse without a jump."""
  if value * last < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _path_values(path) -> tuple[bool, float, float, float]:
  if path is None or not bool(getattr(path, "valid", False)):
    return False, 0.0, 0.0, 0.0
  values = (
    getattr(path, "pathOffset", 0.0),
    getattr(path, "pathAngle", 0.0),
    getattr(path, "curvatureRate", 0.0),
  )
  if not all(math.isfinite(value) for value in values):
    return False, 0.0, 0.0, 0.0
  return True, *(float(value) for value in values)


def _stronger_same_direction(first: float, second: float) -> float:
  if first * second > 0.0:
    return first if abs(first) >= abs(second) else second
  return first


def _target_is_behind_wheel(target: float, measured_curvature: float) -> bool:
  """True only when the target asks to leave the wheel's delivered arc."""
  return target * measured_curvature <= 0.0 or \
         abs(target) + PATH_UNWIND_ERROR_DEADZONE < abs(measured_curvature)


def _undertracking_correction(target: float, measured_curvature: float,
                              projected_curvature: float, limit: float) -> float:
  """Bounded correction after discounting wheel motion already closing target."""
  tracking_error = projected_tracking_error(target, measured_curvature, projected_curvature)
  if tracking_error * target <= 0.0:
    return 0.0
  return _clip(
    _deadzone(tracking_error, PATH_TRACKING_ERROR_DEADZONE),
    (-limit, limit),
  )


def _projected_c0_correction(target: float, measured_curvature: float,
                             projected_curvature: float, limit: float) -> float:
  """Preserve the proven feedback floor and project only added C0 authority."""
  base_correction = _undertracking_correction(
    target, measured_curvature, measured_curvature, PATH_C1_TRACKING_ERROR_LIMIT,
  )
  projected_correction = _undertracking_correction(
    target, measured_curvature, projected_curvature, limit,
  )
  return _stronger_same_direction(base_correction, projected_correction)


def _command_equivalent_curvature(command: LateralPathCommand, distance: float) -> float:
  path_offset = command.path_offset + command.path_angle * distance + \
                0.5 * command.curvature * distance ** 2 + command.curvature_rate * distance ** 3 / 6.0
  return 2.0 * path_offset / distance ** 2


def _preserve_model_direction(path_offset: float, path_angle: float, curvature: float,
                              curvature_rate: float, model_curvature: float) -> float:
  """Use C0 to keep a meaningful model path from crossing through zero."""
  command = LateralPathCommand(
    valid=True,
    path_offset=path_offset,
    path_angle=path_angle,
    curvature=curvature,
    curvature_rate=curvature_rate,
  )
  command_curvature = _command_equivalent_curvature(command, PATH_MIN_LOOKAHEAD)
  if model_curvature * command_curvature >= 0.0:
    return path_offset

  guarded_curvature = math.copysign(PATH_TRACKING_ERROR_DEADZONE, model_curvature)
  correction = 0.5 * (guarded_curvature - command_curvature) * PATH_MIN_LOOKAHEAD ** 2
  return _clip(path_offset + correction, PATH_C0_LIMITS)


def _preview_conflict_share(model_target: float, desired_curvature: float,
                            desired_angle_curvature: float, measured_curvature: float,
                            previous_command_curvature: float, v_ego: float) -> float:
  """Continuously reject preview that fights a delivered-wheel unwind."""
  desired_angle_error = desired_angle_curvature - measured_curvature
  wheel_follows_preview = measured_curvature * model_target > 0.0
  angle_error_opposes_preview = desired_angle_error * model_target < 0.0
  command_has_begun_release = previous_command_curvature * model_target < 0.0
  if not wheel_follows_preview or not angle_error_opposes_preview or not command_has_begun_release:
    return 0.0
  angle_conflict_share = _interp(abs(desired_angle_error), *PATH_PREVIEW_CONFLICT_BP, 0.0, 1.0)
  action_release_share = _interp(abs(desired_curvature), *PATH_PREVIEW_ACTION_RELEASE_BP, 1.0, 0.0)
  released_command_share = _interp(
    abs(previous_command_curvature), *PATH_PREVIEW_RELEASE_COMMAND_BP, 0.0, 1.0,
  )
  moving_confidence = _interp(v_ego, *PATH_PREVIEW_CONFLICT_SPEED_BP, 0.0, 1.0)
  return angle_conflict_share * action_release_share * released_command_share * moving_confidence


def _spatial_unwind_compatibility_share(curvature_rate: float, desired_angle_curvature: float,
                                        measured_curvature: float) -> float:
  """Fade C3 unwind while the delivered wheel still undertracks the angle target."""
  action_is_small = abs(desired_angle_curvature) < PATH_C3_UNWIND_TARGET_MIN
  c3_deepens_action = curvature_rate * desired_angle_curvature >= 0.0
  if action_is_small or c3_deepens_action:
    return 1.0

  tracking_error = desired_angle_curvature - measured_curvature
  wheel_reached_target = tracking_error * desired_angle_curvature <= 0.0
  if wheel_reached_target:
    return 1.0

  conflict_share = _interp(abs(tracking_error), *PATH_C3_UNWIND_ERROR_BP, 0.0, 1.0)
  return 1.0 - conflict_share


class LatControlPath:
  """Map action, path preview, and measured wheel curvature to one polynomial.

  The previous command is the only persistent controller state. The action head
  carries gentle curvature. Coherent maneuver geometry owns C0/C1 until both
  the model and action have moved inside the wheel's delivered arc. A bounded
  model-relative correction adds only authority the wheel has not delivered.
  """

  def __init__(self):
    self._last_command = LateralPathCommand()

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None) -> LateralPathCommand:
    desired_curvature = _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0
    measured_curvature = _finite(measured_curvature)
    projected_feedback_available = projected_measured_curvature is not None
    if projected_measured_curvature is None:
      projected_measured_curvature = measured_curvature
    projected_measured_curvature = _finite(projected_measured_curvature)
    desired_angle_feedback_available = desired_angle_curvature is not None
    if desired_angle_curvature is None:
      desired_angle_curvature = desired_curvature
    desired_angle_curvature = _finite(desired_angle_curvature)
    v_ego = max(_finite(v_ego), 0.0)

    if not active:
      self._last_command = LateralPathCommand()
      return self._last_command

    lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
    valid, path_offset, path_angle, spatial_curvature_rate = _path_values(path)
    if not valid:
      path_offset = 0.0
      path_angle = 0.0
      spatial_curvature_rate = 0.0

    if driver_override:
      command = LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured_curvature * PATH_MIN_LOOKAHEAD ** 2, PATH_C0_LIMITS),
        path_angle=_clip(measured_curvature * lookahead, PATH_C1_LIMITS),
      )
      self._last_command = command
      return command

    offset_curvature = 2.0 * path_offset / PATH_MIN_LOOKAHEAD ** 2 if valid else 0.0
    angle_curvature = path_angle / lookahead if valid else 0.0
    previous_command_curvature = _command_equivalent_curvature(self._last_command, PATH_MIN_LOOKAHEAD)
    model_action_disagreement = False
    preview_conflict_share = 0.0
    geometry_magnitude = min(abs(offset_curvature), abs(angle_curvature))
    geometry_demand = max(abs(offset_curvature), abs(angle_curvature))
    # C0 placement and C1 heading remain coherent when they agree on direction.
    # Requiring both views to clear the maneuver threshold created a turn-exit
    # cliff: one fading term could discard the complete, still-valid polynomial.
    geometry_is_coherent = valid and offset_curvature * angle_curvature > 0.0
    model_command = LateralPathCommand(
      valid=valid,
      path_offset=path_offset,
      path_angle=path_angle,
      curvature=desired_curvature,
      curvature_rate=spatial_curvature_rate,
    )
    model_equivalent_curvature = _command_equivalent_curvature(model_command, PATH_MIN_LOOKAHEAD)
    geometry_equivalent_curvature = offset_curvature + 2.0 * path_angle / PATH_MIN_LOOKAHEAD
    desired_angle_error = desired_angle_curvature - measured_curvature
    strong_angle_rejection = desired_angle_feedback_available and \
                             desired_angle_error * model_equivalent_curvature < 0.0 and \
                             abs(desired_angle_error) >= PATH_PREVIEW_CONFLICT_BP[1] and \
                             v_ego >= PATH_PREVIEW_CONFLICT_SPEED_BP[1]
    delivered_model_exit = geometry_is_coherent and \
                           abs(geometry_equivalent_curvature) > PATH_MODEL_MANEUVER_MIN and \
                           model_equivalent_curvature * geometry_equivalent_curvature > 0.0 and \
                           measured_curvature * geometry_equivalent_curvature > 0.0 and \
                           abs(measured_curvature) >= PATH_TRACKING_ERROR_DEADZONE and \
                           not strong_angle_rejection
    action_opposes_geometry = offset_curvature * desired_curvature < 0.0
    if geometry_is_coherent:
      model_geometry_demand = max(geometry_demand, abs(geometry_equivalent_curvature)) if delivered_model_exit else geometry_demand
      model_geometry_share = 1.0 if action_opposes_geometry else _interp(
        model_geometry_demand, PATH_MODEL_MANEUVER_MIN, PATH_MODEL_GEOMETRY_FULL, 0.0, 1.0,
      )
    else:
      model_geometry_share = 0.0
    coherent_model_maneuver = model_geometry_share > 0.0
    if coherent_model_maneuver:
      raw_model_curvature = geometry_equivalent_curvature if delivered_model_exit else \
                            math.copysign(geometry_magnitude, offset_curvature)
      geometry_reference = desired_curvature if raw_model_curvature * desired_curvature >= 0.0 else 0.0
      raw_model_target = _blend(geometry_reference, raw_model_curvature, model_geometry_share)
      if desired_angle_feedback_available:
        preview_conflict_share = _preview_conflict_share(
          raw_model_curvature, desired_curvature, desired_angle_curvature, measured_curvature,
          previous_command_curvature, v_ego,
        )
      # C0 lateral placement and C1 heading are independent polynomial
      # geometry. Introduce low-amplitude preview continuously, then preserve
      # both instead of collapsing them to the weaker view.
      # When delivered wheel motion and desired angle agree that the preview is
      # stale, continuously fade the complete preview toward the angle target.
      offset_model_target = _blend(geometry_reference, offset_curvature, model_geometry_share)
      angle_model_target = _blend(geometry_reference, angle_curvature, model_geometry_share)
      offset_model_target = _blend(offset_model_target, desired_angle_curvature, preview_conflict_share)
      angle_model_target = _blend(angle_model_target, desired_angle_curvature, preview_conflict_share)
      model_target = _blend(raw_model_target, desired_angle_curvature, preview_conflict_share)
      angle_target = _stronger_same_direction(angle_model_target, desired_curvature)
      action_reversal = abs(desired_curvature) >= PATH_MODEL_MANEUVER_MIN and \
                        model_target * desired_curvature < 0.0
      wheel_reversal = abs(measured_curvature) >= PATH_MODEL_MANEUVER_MIN and \
                       model_target * measured_curvature < 0.0
      # Even sub-threshold action sign noise is enough to keep C0 on the
      # conservative correction bound. It must not promote the model preview
      # to the larger same-direction tracking correction.
      model_action_disagreement = model_target * desired_curvature < 0.0 or wheel_reversal
      if action_reversal or wheel_reversal:
        offset_target = offset_model_target
      else:
        c0_model_share = _interp(abs(desired_curvature), 0.003, 0.006, 0.0, 1.0)
        offset_target = desired_curvature + c0_model_share * (offset_model_target - desired_curvature)
    elif valid:
      model_target = desired_curvature
      offset_target = desired_curvature
      angle_target = desired_curvature
    else:
      model_target = 0.0
      offset_target = 0.0
      angle_target = 0.0

    wheel_beyond_action = _target_is_behind_wheel(desired_curvature, measured_curvature)
    wheel_beyond_model = not coherent_model_maneuver or _target_is_behind_wheel(model_target, measured_curvature)
    wheel_beyond_target = wheel_beyond_action and wheel_beyond_model

    # Once the wheel is beyond the action target, actively remove the old turn.
    # Model geometry must independently agree that the delivered arc is stale;
    # a falling action alone cannot release a turn the model still requires.
    if wheel_beyond_target:
      offset_target += _clip(
        _deadzone(offset_target - measured_curvature, PATH_UNWIND_ERROR_DEADZONE),
        (-PATH_UNWIND_LIMIT, PATH_UNWIND_LIMIT),
      )
      angle_target += _clip(
        _deadzone(angle_target - measured_curvature, PATH_UNWIND_ERROR_DEADZONE),
        (-PATH_UNWIND_LIMIT, PATH_UNWIND_LIMIT),
      )
    else:
      # C0 is the fast placement servo. Project only its authority above the
      # established C1-sized feedback floor, so measured motion can prevent an
      # extra placement shove without weakening the heading command.
      offset_target += _projected_c0_correction(
        offset_model_target if coherent_model_maneuver else model_target,
        measured_curvature, projected_measured_curvature,
        PATH_C1_TRACKING_ERROR_LIMIT if model_action_disagreement else PATH_C0_TRACKING_ERROR_LIMIT,
      )
      angle_target += _undertracking_correction(
        angle_model_target if coherent_model_maneuver else model_target,
        measured_curvature, measured_curvature,
        PATH_C1_TRACKING_ERROR_LIMIT,
      )

    c2_action_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 1.0, 0.0)
    action_tracking_error = desired_curvature - measured_curvature
    unresolved = max(abs(desired_curvature), abs(measured_curvature), abs(action_tracking_error))
    c2_settled_share = _interp(unresolved, *PATH_C2_SETTLED_BP, 1.0, 0.0)
    c2_share = min(c2_action_share, c2_settled_share)
    allocated_c2 = _clip(desired_curvature * c2_share, PATH_C2_LIMITS)
    curvature = allocated_c2
    curvature = _limit_attack(curvature, self._last_command.curvature, PATH_C2_SLEW)

    maneuver_demand = max(abs(desired_curvature), abs(offset_curvature), abs(angle_curvature))
    c0_share = 1.0 if wheel_beyond_target else _interp(maneuver_demand, 0.003, 0.006, 0.0, 1.0)
    if valid:
      path_offset_command = _clip(
        0.5 * (offset_target - allocated_c2) * PATH_MIN_LOOKAHEAD ** 2 * c0_share,
        PATH_C0_LIMITS,
      )
      path_angle_command = _clip((angle_target - allocated_c2) * lookahead, PATH_C1_LIMITS)
    else:
      path_offset_command = 0.0
      path_angle_command = 0.0

    c3_action_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 0.0, 1.0)
    coherent_reversal = measured_curvature * angle_curvature < 0.0 and \
                        spatial_curvature_rate * angle_curvature > 0.0 and \
                        offset_curvature * angle_curvature > 0.0
    c3_preview_share = _interp(maneuver_demand, *PATH_C2_FADE_BP, 0.0, 1.0) if coherent_reversal else 0.0
    c3_share = max(c3_action_share, c3_preview_share)
    curvature_rate_command = _clip(spatial_curvature_rate * c3_share, PATH_C3_LIMITS)
    curvature_rate_command *= 1.0 - preview_conflict_share
    if desired_angle_feedback_available:
      compatibility_target = desired_angle_curvature
      compatibility_wheel = measured_curvature
      if projected_feedback_available:
        # C3 is d(curvature)/distance. Project its moving target over the same
        # horizon as the wheel so unwind is faded before the wheel undershoots.
        compatibility_target += curvature_rate_command * v_ego * FORD_PATH_ANGLE_PROJECTION_HORIZON
        compatibility_wheel = projected_measured_curvature
      curvature_rate_command *= _spatial_unwind_compatibility_share(
        curvature_rate_command, compatibility_target, compatibility_wheel,
      )

    path_offset_command = _limit_attack(
      path_offset_command,
      self._last_command.path_offset,
      0.5 * PATH_MANEUVER_CURVATURE_SLEW * PATH_MIN_LOOKAHEAD ** 2,
    )
    path_angle_command = _limit_attack(
      path_angle_command,
      self._last_command.path_angle,
      PATH_MANEUVER_CURVATURE_SLEW * lookahead,
    )
    curvature_rate_command = _limit_attack(
      curvature_rate_command,
      self._last_command.curvature_rate,
      PATH_C3_SLEW,
    )

    meaningful_model_geometry = delivered_model_exit and preview_conflict_share < 1.0
    if meaningful_model_geometry:
      path_offset_command = _preserve_model_direction(
        path_offset_command, path_angle_command, curvature, curvature_rate_command,
        model_equivalent_curvature,
      )

    command = LateralPathCommand(
      valid=valid,
      path_offset=path_offset_command,
      path_angle=path_angle_command,
      curvature=curvature,
      curvature_rate=curvature_rate_command,
    )
    self._last_command = command
    return command
