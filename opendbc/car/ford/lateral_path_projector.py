"""Coherent Ford LMC2 polynomial controller."""

from __future__ import annotations

from dataclasses import dataclass
import math


PATH_LIMITS = (
  (-4.61, 4.60),
  (-0.475, 0.497),
  (-0.02, 0.02),
  (-0.001024, 0.001023),
)
PATH_MIN_LOOKAHEAD = 7.0
PATH_MANEUVER_CURVATURE_SLEW = 0.006
PATH_C2_SLEW = 0.0002
PATH_C3_SLEW = 0.0002
PATH_C2_FADE_BP = (0.006, 0.012)
PATH_C2_SETTLED_BP = (0.003, 0.006)
PATH_PREVIEW_BP = (0.003, 0.012)
PATH_C0_BP = (0.003, 0.006)
PATH_TRACKING_ERROR_DEADZONE = 0.0005
PATH_C0_TRACKING_ERROR_LIMIT = 0.02
PATH_C1_TRACKING_ERROR_LIMIT = 0.012
PATH_UNWIND_ERROR_DEADZONE = 0.0005
PATH_UNWIND_LIMIT = 0.006
PATH_C3_UNWIND_ERROR_BP = (0.0005, 0.002)
PATH_C3_UNWIND_TARGET_MIN = 0.003
PATH_DIRECTION_MARGIN = 0.0005


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0

  def coefficients(self) -> tuple[float, float, float, float]:
    return self.path_offset, self.path_angle, self.curvature, self.curvature_rate


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


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
  if value * last < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _basis(distance: float) -> tuple[float, float, float, float]:
  return 2.0 / distance ** 2, 2.0 / distance, 1.0, distance / 3.0


def _attack_bounds(last: float, step: float, limits: tuple[float, float]) -> tuple[float, float]:
  if last > 0.0:
    return max(limits[0], -step), min(limits[1], last + step)
  if last < 0.0:
    return max(limits[0], last - step), min(limits[1], step)
  return max(limits[0], -step), min(limits[1], step)


def _equivalent_curvature(coefficients: tuple[float, float, float, float], distance: float = 7.0) -> float:
  basis = _basis(distance)
  return sum(basis[i] * coefficients[i] for i in range(4))


def _target_is_behind_wheel(target: float, measured_curvature: float) -> bool:
  """Whether the target asks to leave the wheel's currently delivered arc."""
  return target * measured_curvature <= 0.0 or \
         abs(target) + PATH_UNWIND_ERROR_DEADZONE < abs(measured_curvature)


def _undertracking_correction(target: float, measured_curvature: float, limit: float) -> float:
  """Add authority only while the delivered wheel remains behind the target."""
  tracking_error = target - measured_curvature
  if tracking_error * target <= 0.0:
    return 0.0
  return _clip(
    _deadzone(tracking_error, PATH_TRACKING_ERROR_DEADZONE),
    (-limit, limit),
  )


def _unwind_target(target: float, measured_curvature: float) -> float:
  """Move a delivered target toward zero without crossing its model direction."""
  corrected = target + _clip(
    _deadzone(target - measured_curvature, PATH_UNWIND_ERROR_DEADZONE),
    (-PATH_UNWIND_LIMIT, PATH_UNWIND_LIMIT),
  )
  return 0.0 if corrected * target < 0.0 else corrected


def _compose_path_target(raw_target: tuple[float, float, float, float],
                         measured_curvature: float, desired_angle_curvature: float,
                         v_ego: float, valid: bool,
                         allocated_c2: float, allocated_c3: float) \
                         -> tuple[tuple[float, float, float, float], float, bool]:
  """Resolve model samples and action into one non-duplicated Ford polynomial.

  pathOffset and pathAngle are independent observations of the model trajectory,
  while curvature and curvatureRate are action and slope. Convert the first two
  to curvature observations, resolve one current-frame intent, and allocate C2
  exactly once before encoding the remaining maneuver authority into C0/C1.
  """
  lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
  desired_curvature = raw_target[2]
  offset_curvature = 2.0 * raw_target[0] / PATH_MIN_LOOKAHEAD ** 2 if valid else desired_curvature
  angle_curvature = raw_target[1] / lookahead if valid else desired_curvature
  geometry_demand = max(abs(offset_curvature), abs(angle_curvature))
  geometry_is_coherent = valid and offset_curvature * angle_curvature > 0.0

  if geometry_is_coherent:
    geometry_share = _interp(geometry_demand, *PATH_PREVIEW_BP, 0.0, 1.0)
    geometry_curvature = math.copysign(
      min(abs(offset_curvature), abs(angle_curvature)),
      offset_curvature,
    )
    geometry_reference = desired_curvature if geometry_curvature * desired_curvature >= 0.0 else 0.0
    model_target = _blend(geometry_reference, geometry_curvature, geometry_share)
    offset_target = _blend(geometry_reference, offset_curvature, geometry_share)
    angle_target = _blend(geometry_reference, angle_curvature, geometry_share)
  else:
    geometry_share = 0.0
    model_target = desired_curvature
    offset_target = desired_curvature
    angle_target = desired_curvature

  stale_model_geometry = geometry_share > 0.0 and \
                         model_target * raw_target[3] < 0.0 and \
                         model_target * desired_angle_curvature < 0.0 and \
                         measured_curvature * model_target > 0.0
  if stale_model_geometry:
    geometry_share = 0.0
    model_target = desired_angle_curvature
    offset_target = desired_angle_curvature
    angle_target = desired_angle_curvature

  coherent_model_maneuver = geometry_share > 0.0
  wheel_beyond_action = _target_is_behind_wheel(desired_curvature, measured_curvature)
  wheel_beyond_model = not coherent_model_maneuver or \
                       _target_is_behind_wheel(model_target, measured_curvature)
  wheel_beyond_target = wheel_beyond_action and wheel_beyond_model

  if wheel_beyond_target:
    offset_target = _unwind_target(offset_target, measured_curvature)
    angle_target = _unwind_target(angle_target, measured_curvature)
  else:
    model_action_disagreement = model_target * desired_curvature < 0.0 or \
                                model_target * measured_curvature < 0.0
    offset_target += _undertracking_correction(
      offset_target,
      measured_curvature,
      PATH_C1_TRACKING_ERROR_LIMIT if model_action_disagreement else PATH_C0_TRACKING_ERROR_LIMIT,
    )
    angle_target += _undertracking_correction(
      angle_target,
      measured_curvature,
      PATH_C1_TRACKING_ERROR_LIMIT,
    )

  maneuver_demand = max(abs(desired_curvature), geometry_demand)
  c0_share = 1.0 if wheel_beyond_target else \
             _interp(maneuver_demand, *PATH_C0_BP, 0.0, 1.0)
  offset_residual = offset_target - allocated_c2
  angle_residual = angle_target - allocated_c2
  if offset_target * allocated_c2 > 0.0 and abs(offset_target) < abs(allocated_c2):
    offset_residual = 0.0
  if angle_target * allocated_c2 > 0.0 and abs(angle_target) < abs(allocated_c2):
    angle_residual = 0.0
  target = (
    0.5 * offset_residual * PATH_MIN_LOOKAHEAD ** 2 * c0_share,
    angle_residual * lookahead,
    allocated_c2,
    allocated_c3,
  )
  preserve_model_direction = coherent_model_maneuver
  return target, model_target, preserve_model_direction


def _preserve_model_direction(coefficients: tuple[float, float, float, float],
                              bounds: tuple[tuple[float, float], ...],
                              model_curvature: float) -> tuple[float, float, float, float]:
  command_curvature = _equivalent_curvature(coefficients)
  if model_curvature * command_curvature >= 0.0:
    return coefficients

  guarded_curvature = math.copysign(PATH_DIRECTION_MARGIN, model_curvature)
  values = list(coefficients)
  basis = _basis(PATH_MIN_LOOKAHEAD)
  for i in (0, 1):
    correction = (guarded_curvature - command_curvature) / basis[i]
    values[i] = _clip(values[i] + correction, bounds[i])
    command_curvature = _equivalent_curvature(tuple(values))
    if model_curvature * command_curvature >= 0.0:
      return tuple(values)

  return 0.0, 0.0, 0.0, 0.0


def _c3_compatibility_share(curvature_rate: float, desired_curvature: float,
                            projected_curvature: float) -> float:
  if abs(desired_curvature) < PATH_C3_UNWIND_TARGET_MIN or curvature_rate * desired_curvature >= 0.0:
    return 1.0

  tracking_error = desired_curvature - projected_curvature
  if tracking_error * desired_curvature <= 0.0:
    return 1.0
  conflict_share = _interp(abs(tracking_error), *PATH_C3_UNWIND_ERROR_BP, 0.0, 1.0)
  return 1.0 - conflict_share


class ProjectedLatControlPath:
  """Return one coherent, bounded Ford polynomial through a stable interface."""

  def __init__(self):
    self._last_command = LateralPathCommand()

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None) -> LateralPathCommand:
    measured_curvature = _finite(measured_curvature)
    projected_measured_curvature = measured_curvature if projected_measured_curvature is None else \
                                   _finite(projected_measured_curvature, measured_curvature)
    v_ego = max(_finite(v_ego), 0.0)

    if not active:
      self._last_command = LateralPathCommand()
      return self._last_command

    valid = path is not None and bool(getattr(path, "valid", False))
    if not valid:
      target = (0.0, 0.0, _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0, 0.0)
    else:
      target = (
        _finite(getattr(path, "pathOffset", 0.0)),
        _finite(getattr(path, "pathAngle", 0.0)),
        _finite(getattr(path, "curvature", 0.0)),
        _finite(getattr(path, "curvatureRate", 0.0)),
      )
    desired_angle_curvature = target[2] if desired_angle_curvature is None else _finite(desired_angle_curvature, target[2])

    if driver_override:
      command = LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured_curvature * PATH_MIN_LOOKAHEAD ** 2, PATH_LIMITS[0]),
        path_angle=_clip(measured_curvature * max(v_ego, PATH_MIN_LOOKAHEAD), PATH_LIMITS[1]),
      )
      self._last_command = command
      return command

    raw_target = target
    lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)

    attack_steps = (
      0.5 * PATH_MANEUVER_CURVATURE_SLEW * PATH_MIN_LOOKAHEAD ** 2,
      PATH_MANEUVER_CURVATURE_SLEW * lookahead,
      PATH_C2_SLEW,
      PATH_C3_SLEW,
    )
    bounds = [
      _attack_bounds(last, step, limits)
      for last, step, limits in zip(self._last_command.coefficients(), attack_steps, PATH_LIMITS, strict=True)
    ]
    action_tracking_error = raw_target[2] - measured_curvature
    unresolved = max(abs(raw_target[2]), abs(action_tracking_error))
    c2_share = min(
      _interp(abs(raw_target[2]), *PATH_C2_FADE_BP, 1.0, 0.0),
      _interp(unresolved, *PATH_C2_SETTLED_BP, 1.0, 0.0),
    )
    safe_c2 = _limit_attack(_clip(raw_target[2] * c2_share, PATH_LIMITS[2]),
                            self._last_command.curvature, PATH_C2_SLEW)
    bounds[2] = (safe_c2, safe_c2)
    c3_share = _c3_compatibility_share(raw_target[3], desired_angle_curvature, projected_measured_curvature)
    safe_c3 = _limit_attack(_clip(raw_target[3] * c3_share, PATH_LIMITS[3]),
                            self._last_command.curvature_rate, PATH_C3_SLEW)
    bounds[3] = (safe_c3, safe_c3)

    target, model_curvature, preserve_model_direction = _compose_path_target(
      raw_target, measured_curvature, desired_angle_curvature,
      v_ego, valid, safe_c2, safe_c3,
    )
    coefficient_bounds = tuple(bounds)
    coefficients = tuple(
      _clip(value, bound)
      for value, bound in zip(target, coefficient_bounds, strict=True)
    )
    if preserve_model_direction:
      coefficients = _preserve_model_direction(coefficients, coefficient_bounds, model_curvature)
    command = LateralPathCommand(valid=valid, path_offset=coefficients[0], path_angle=coefficients[1],
                                 curvature=coefficients[2], curvature_rate=coefficients[3])
    self._last_command = command
    return command
