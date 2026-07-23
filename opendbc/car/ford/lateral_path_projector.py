"""Constrained Ford LMC2 polynomial projection controller."""

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
PATH_PROJECTION_DISTANCES = (3.0, 7.0, 15.0, 30.0)
PATH_PROJECTION_SCALES = (4.6, 0.5, 0.02, 0.001)
PATH_MANEUVER_CURVATURE_SLEW = 0.006
PATH_C2_SLEW = 0.0002
PATH_C3_SLEW = 0.0002
PATH_PROJECTION_ITERATIONS = 96
PATH_C2_FADE_BP = (0.006, 0.012)
PATH_C2_SETTLED_BP = (0.003, 0.006)
PATH_PREVIEW_BP = (0.003, 0.012)
PATH_C3_UNWIND_ERROR_BP = (0.0005, 0.002)
PATH_C3_UNWIND_TARGET_MIN = 0.003
PATH_MODEL_MANEUVER_MIN = 0.003
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


def _limit_attack(value: float, last: float, max_step: float) -> float:
  if value * last < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _basis(distance: float) -> tuple[float, float, float, float]:
  return 2.0 / distance ** 2, 2.0 / distance, 1.0, distance / 3.0


PROJECTION_BASIS = tuple(_basis(distance) for distance in PATH_PROJECTION_DISTANCES)


def _attack_bounds(last: float, step: float, limits: tuple[float, float]) -> tuple[float, float]:
  if last > 0.0:
    return max(limits[0], -step), min(limits[1], last + step)
  if last < 0.0:
    return max(limits[0], last - step), min(limits[1], step)
  return max(limits[0], -step), min(limits[1], step)


def _project_coefficients(target: tuple[float, float, float, float],
                          bounds: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
  """Solve a four-variable box-constrained polynomial least-squares problem."""
  scaled_basis = tuple(
    tuple(row[i] * PATH_PROJECTION_SCALES[i] for i in range(4))
    for row in PROJECTION_BASIS
  )
  target_samples = tuple(sum(row[i] * target[i] for i in range(4)) for row in PROJECTION_BASIS)
  hessian = tuple(
    tuple(sum(row[i] * row[j] for row in scaled_basis) for j in range(4))
    for i in range(4)
  )
  gradient = tuple(
    sum(scaled_basis[row][i] * target_samples[row] for row in range(4))
    for i in range(4)
  )
  scaled_bounds = tuple(
    (bounds[i][0] / PATH_PROJECTION_SCALES[i], bounds[i][1] / PATH_PROJECTION_SCALES[i])
    for i in range(4)
  )
  values = [
    _clip(target[i] / PATH_PROJECTION_SCALES[i], scaled_bounds[i])
    for i in range(4)
  ]

  for _ in range(PATH_PROJECTION_ITERATIONS):
    for i in range(4):
      residual = gradient[i] - sum(hessian[i][j] * values[j] for j in range(4) if j != i)
      values[i] = _clip(residual / hessian[i][i], scaled_bounds[i])

  return tuple(values[i] * PATH_PROJECTION_SCALES[i] for i in range(4))


def _equivalent_curvature(coefficients: tuple[float, float, float, float], distance: float = 7.0) -> float:
  basis = _basis(distance)
  return sum(basis[i] * coefficients[i] for i in range(4))


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


def _shape_preserving_delivery_share(model_path: tuple[float, float, float, float],
                                     command_curvature: float, projected_curvature: float,
                                     desired_curvature: float) -> float:
  """Scale the full polynomial for wrong-direction, overtracking, and spatial unwind."""
  model_curvature = _equivalent_curvature(model_path)
  reference_curvature = model_curvature
  if reference_curvature == 0.0:
    reference_curvature = desired_curvature if desired_curvature != 0.0 else projected_curvature
  if reference_curvature == 0.0:
    return 1.0

  direction = math.copysign(1.0, reference_curvature)
  command_along = direction * command_curvature
  measured_along = direction * projected_curvature
  desired_along = direction * desired_curvature
  if command_along <= 0.0:
    return 0.0 if abs(model_curvature) >= PATH_MODEL_MANEUVER_MIN else 1.0
  if measured_along <= 0.0 or desired_along >= measured_along:
    return 1.0

  delivery_share = 1.0
  if command_along > measured_along:
    delivery_share = _clip(measured_along / command_along, (0.0, 1.0))

  near_curvature = direction * _equivalent_curvature(model_path, PATH_PROJECTION_DISTANCES[0])
  far_curvature = direction * _equivalent_curvature(model_path, PATH_PROJECTION_DISTANCES[2])
  path_scale = max(abs(near_curvature), abs(far_curvature))
  spatial_unwind = _clip((near_curvature - far_curvature) / path_scale, (0.0, 1.0)) if path_scale > 0.0 else 0.0
  desired_share = _clip(max(desired_along, 0.0) / command_along, (0.0, 1.0))
  unwind_share = 1.0 - spatial_unwind * (1.0 - desired_share)
  return min(delivery_share, unwind_share)


class ProjectedLatControlPath:
  """Return the closest feasible Ford polynomial through one stable interface."""

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
    model_curvature = _equivalent_curvature(raw_target)
    lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
    offset_curvature = 2.0 * raw_target[0] / PATH_MIN_LOOKAHEAD ** 2
    angle_curvature = raw_target[1] / lookahead
    geometry_curvature = offset_curvature + 2.0 * raw_target[1] / PATH_MIN_LOOKAHEAD
    delivered_model_geometry = valid and offset_curvature * angle_curvature > 0.0 and \
                               abs(geometry_curvature) > PATH_MODEL_MANEUVER_MIN and \
                               model_curvature * geometry_curvature > 0.0 and \
                               measured_curvature * geometry_curvature > 0.0 and \
                               abs(measured_curvature) >= PATH_DIRECTION_MARGIN
    if valid:
      offset_curvature = 2.0 * target[0] / PATH_MIN_LOOKAHEAD ** 2
      angle_curvature = target[1] / lookahead
      preview_share = _interp(max(abs(offset_curvature), abs(angle_curvature)),
                              *PATH_PREVIEW_BP, 0.0, 1.0)
      target = (target[0] * preview_share, target[1] * preview_share, target[2], target[3])

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
    action_tracking_error = target[2] - measured_curvature
    unresolved = max(abs(target[2]), abs(measured_curvature), abs(action_tracking_error))
    c2_share = min(
      _interp(abs(target[2]), *PATH_C2_FADE_BP, 1.0, 0.0),
      _interp(unresolved, *PATH_C2_SETTLED_BP, 1.0, 0.0),
    )
    safe_c2 = _limit_attack(_clip(target[2] * c2_share, PATH_LIMITS[2]),
                            self._last_command.curvature, PATH_C2_SLEW)
    bounds[2] = (safe_c2, safe_c2)
    c3_share = _c3_compatibility_share(target[3], desired_angle_curvature, projected_measured_curvature)
    safe_c3 = _limit_attack(_clip(target[3] * c3_share, PATH_LIMITS[3]),
                            self._last_command.curvature_rate, PATH_C3_SLEW)
    bounds[3] = (safe_c3, safe_c3)

    coefficient_bounds = tuple(bounds)
    coefficients = _project_coefficients(target, coefficient_bounds)
    if delivered_model_geometry:
      coefficients = _preserve_model_direction(coefficients, coefficient_bounds, model_curvature)
    delivery_share = _shape_preserving_delivery_share(
      raw_target, _equivalent_curvature(coefficients),
      projected_measured_curvature, desired_angle_curvature,
    )
    coefficients = tuple(coefficient * delivery_share for coefficient in coefficients)
    command = LateralPathCommand(valid=valid, path_offset=coefficients[0], path_angle=coefficients[1],
                                 curvature=coefficients[2], curvature_rate=coefficients[3])
    self._last_command = command
    return command
