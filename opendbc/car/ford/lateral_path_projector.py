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
PATH_C2_SLEW = 0.0002
PATH_C3_SLEW = 0.0002
PATH_C2_BASEBAND_BP = (0.003, 0.006)
PATH_C2_SPATIAL_DELTA_BP = (0.0015, 0.0045)
PATH_ACTION_SUPPORT_BP = (0.1, 0.2)
PATH_PREVIEW_BP = (0.003, 0.012)
PATH_TRACKING_ERROR_DEADZONE = 0.0005
PATH_C0_TRACKING_ERROR_LIMIT = 0.02
PATH_C0_CONTINUATION_ERROR_LIMIT = 0.04
PATH_C0_CONTINUATION_MARGIN = 0.006
PATH_ORDINARY_ARRIVAL_MARGIN = 0.00025
PATH_C1_TRACKING_ERROR_LIMIT = 0.012
PATH_UNWIND_ERROR_DEADZONE = 0.0005
PATH_UNWIND_LIMIT = 0.006
PATH_PROJECTED_ARRIVAL_ERROR_BP = (0.0005, 0.002)
PATH_C3_UNWIND_ERROR_BP = PATH_PROJECTED_ARRIVAL_ERROR_BP
PATH_MEASURED_ARRIVAL_ERROR_BP = PATH_PROJECTED_ARRIVAL_ERROR_BP
PATH_DIRECTION_MARGIN = 0.0005
PATH_C3_TO_C0_LOOKAHEAD = 15.0


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0

  def coefficients(self) -> tuple[float, float, float, float]:
    return self.path_offset, self.path_angle, self.curvature, self.curvature_rate


PathCoefficients = tuple[float, float, float, float]
PathBounds = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _PathFrame:
  """All immutable observations and targets for one controller update."""

  valid: bool
  raw_target: PathCoefficients
  measured_curvature: float
  projected_curvature: float
  desired_curvature: float
  v_ego: float
  lat_ctl_limit: int
  lookahead: float
  offset_curvature: float
  angle_curvature: float
  model_curvature: float
  projected_tracking_error: float


@dataclass(frozen=True)
class _AuthorityPlan:
  """One allocation of the model path across Ford's four coefficients."""

  residual_share: float
  proactive_preview_share: float
  anchored_c2: float
  safe_c3: float
  full_target: PathCoefficients
  model_curvature: float
  preserve_model_direction: bool
  bounds: PathBounds


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _observe_path(
  path, measured_curvature: float, projected_curvature: float | None, desired_curvature: float | None, v_ego: float, lat_ctl_limit: int
) -> _PathFrame:
  """Normalize the caller-facing inputs once for all downstream policy."""
  measured_curvature = _finite(measured_curvature)
  projected_curvature = measured_curvature if projected_curvature is None else _finite(projected_curvature, measured_curvature)
  v_ego = max(_finite(v_ego), 0.0)
  valid = path is not None and bool(getattr(path, "valid", False))
  if valid:
    raw_target = (
      _finite(getattr(path, "pathOffset", 0.0)),
      _finite(getattr(path, "pathAngle", 0.0)),
      _finite(getattr(path, "curvature", 0.0)),
      _finite(getattr(path, "curvatureRate", 0.0)),
    )
  else:
    raw_target = 0.0, 0.0, _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0, 0.0
  desired_curvature = raw_target[2] if desired_curvature is None else _finite(desired_curvature, raw_target[2])
  lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
  tracking_error = _projected_tracking_error(desired_curvature, measured_curvature, projected_curvature)
  return _PathFrame(
    valid,
    raw_target,
    measured_curvature,
    projected_curvature,
    desired_curvature,
    v_ego,
    lat_ctl_limit,
    lookahead,
    2.0 * raw_target[0] / PATH_MIN_LOOKAHEAD**2,
    raw_target[1] / lookahead,
    _equivalent_curvature(raw_target),
    tracking_error,
  )


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


def _apply_c2_attack(value: float, last: float) -> float:
  """Deliver the ordinary C2 band immediately without changing large-turn attack."""
  if value * last < 0.0:
    return _limit_attack(value, last, PATH_C2_SLEW)
  if abs(value) <= PATH_C2_BASEBAND_BP[1]:
    return value
  return _limit_attack(value, last, PATH_C2_SLEW)


def _basis(distance: float) -> tuple[float, float, float, float]:
  return 2.0 / distance**2, 2.0 / distance, 1.0, distance / 3.0


def _equivalent_curvature(coefficients: tuple[float, float, float, float], distance: float = 7.0) -> float:
  basis = _basis(distance)
  return sum(basis[i] * coefficients[i] for i in range(4))


def lmc2_control_utilization(command: LateralPathCommand, lat_ctl_limit: int) -> float:
  """Return signed LMC2 command-envelope usage, augmented by PSCM limit status.

  Coefficient saturation and the PSCM's physical steering limit are not the
  same thing. The coefficient ratios provide a continuous meter, while the
  PSCM status raises the meter near/full only when it reports LimitClose or
  LimitReached. LimitWithDriverActive deliberately does not imply saturation.
  """
  coefficients = command.coefficients()
  coefficient_utilization = [abs(value) / (limits[1] if value >= 0.0 else abs(limits[0])) for value, limits in zip(coefficients, PATH_LIMITS, strict=True)]
  coefficient_magnitude = _clip(max(coefficient_utilization), (0.0, 1.0))
  if coefficient_magnitude == 0.0:
    return 0.0

  magnitude = coefficient_magnitude
  if lat_ctl_limit == 1:  # LimitClose
    magnitude = max(magnitude, 0.8)
  elif lat_ctl_limit == 2:  # LimitReached
    magnitude = 1.0

  direction_source = _equivalent_curvature(coefficients, PATH_MIN_LOOKAHEAD)
  if abs(direction_source) < 1e-9:
    direction_source = coefficients[max(range(4), key=coefficient_utilization.__getitem__)]
  return math.copysign(magnitude, direction_source)


def _preview_allocation(frame: _PathFrame) -> tuple[float, float]:
  """Return maneuver and proactive C0/C1 shares from one model observation."""
  raw_target = frame.raw_target
  desired_curvature = abs(raw_target[2])
  curvature_rate_demand = abs(frame.raw_target[3]) * frame.lookahead / 3.0
  if not frame.valid:
    maneuver_demand = max(curvature_rate_demand, desired_curvature - PATH_LIMITS[2][1], 0.0)
    return _interp(maneuver_demand, *PATH_C2_BASEBAND_BP, 0.0, 1.0), 0.0

  coherent_geometry = frame.offset_curvature * frame.angle_curvature > 0.0
  geometry_demand = min(abs(frame.offset_curvature), abs(frame.angle_curvature)) if coherent_geometry else 0.0
  maneuver_demand = max(curvature_rate_demand, desired_curvature - PATH_LIMITS[2][1], geometry_demand - PATH_LIMITS[2][1], 0.0)
  residual_share = _interp(maneuver_demand, *PATH_C2_BASEBAND_BP, 0.0, 1.0)

  spatial_slope = raw_target[3]
  tracking_error = frame.desired_curvature - frame.measured_curvature
  if frame.model_curvature * frame.desired_curvature > 0.0 and spatial_slope * frame.desired_curvature > 0.0 and tracking_error * frame.desired_curvature > 0.0:
    tracking_share = _interp(abs(tracking_error), *PATH_C2_BASEBAND_BP, 0.0, 1.0)
    action_share = _interp(abs(frame.desired_curvature), *PATH_C2_BASEBAND_BP, 0.0, 1.0)
    spatial_support_share = _interp(abs(spatial_slope) * frame.lookahead / 3.0, 0.0, PATH_PREVIEW_BP[1], 0.0, 1.0)
    residual_share = max(residual_share, tracking_share * action_share * spatial_support_share)

  if (
    not coherent_geometry
    or frame.offset_curvature * spatial_slope <= 0.0
    or raw_target[2] * spatial_slope < 0.0
    or frame.desired_curvature * spatial_slope < 0.0
  ):
    return residual_share, 0.0

  geometry_growth = max(geometry_demand - abs(raw_target[2]), 0.0)
  geometry_share = _interp(geometry_growth, PATH_TRACKING_ERROR_DEADZONE, PATH_C2_BASEBAND_BP[0], 0.0, 1.0)
  slope_share = _interp(abs(spatial_slope) * PATH_MIN_LOOKAHEAD, *PATH_C2_SPATIAL_DELTA_BP, 0.0, 1.0)
  return residual_share, (1.0 - residual_share) * geometry_share * slope_share


def _extend_preview(
  coefficients: PathCoefficients, raw_target: PathCoefficients, direction: float, curvature_budget: float, aggregate_budget: float = math.inf
) -> PathCoefficients:
  """Distribute one curvature budget across unused model C0/C1."""
  basis = _basis(PATH_MIN_LOOKAHEAD)
  values = list(coefficients)
  available = [max((raw_target[i] - values[i]) * direction, 0.0) * basis[i] for i in (0, 1)]
  available_curvature = sum(available)
  if available_curvature == 0.0:
    return coefficients

  used_curvature = min(curvature_budget, available_curvature, aggregate_budget)
  for i in (0, 1):
    if available[i] > 0.0:
      coefficient_delta = direction * used_curvature * available[i] / available_curvature / basis[i]
      values[i] = _clip(values[i] + coefficient_delta, PATH_LIMITS[i])
  return tuple(values)


def _extend_c0_c1_for_proactive_preview(coefficients: PathCoefficients, frame: _PathFrame, preview_share: float) -> PathCoefficients:
  """Add bounded model C0/C1 without borrowing from the C2 anchor."""
  if preview_share == 0.0:
    return coefficients
  return _extend_preview(
    coefficients,
    frame.raw_target,
    math.copysign(1.0, frame.raw_target[3]),
    abs(frame.raw_target[3]) * PATH_MIN_LOOKAHEAD / 3.0 * preview_share,
  )


def _target_is_behind_wheel(target: float, measured_curvature: float) -> bool:
  """Whether the target asks to leave the wheel's currently delivered arc."""
  return target * measured_curvature <= 0.0 or abs(target) + PATH_UNWIND_ERROR_DEADZONE < abs(measured_curvature)


def _undertracking_correction(target: float, measured_curvature: float, limit: float) -> float:
  """Return bounded model-error authority while the wheel is behind target."""
  tracking_error = target - measured_curvature
  if tracking_error * target <= 0.0:
    return 0.0
  return _clip(
    _deadzone(tracking_error, PATH_TRACKING_ERROR_DEADZONE),
    (-limit, limit),
  )


def _projected_tracking_error(target: float, measured_curvature: float, projected_curvature: float) -> float:
  """Return the smaller desired-angle shortfall from current and projected wheel."""
  measured_error = target - measured_curvature
  projected_error = target - projected_curvature
  if measured_error * target <= 0.0 or projected_error * target <= 0.0:
    return 0.0

  return math.copysign(
    min(abs(measured_error), abs(projected_error)),
    target,
  )


def _gated_tracking_correction(model_target: float, desired_angle_target: float, measured_curvature: float, projected_curvature: float, limit: float) -> float:
  """Preserve model authority, but only while desired angle permits correction."""
  desired_error = _projected_tracking_error(
    desired_angle_target,
    measured_curvature,
    projected_curvature,
  )
  desired_correction = _clip(
    _deadzone(desired_error, PATH_TRACKING_ERROR_DEADZONE),
    (-limit, limit),
  )
  if desired_correction == 0.0:
    return 0.0

  model_correction = _undertracking_correction(model_target, measured_curvature, limit)
  if model_correction * desired_correction <= 0.0:
    full_correction = desired_correction
  else:
    full_correction = model_correction if abs(model_correction) >= abs(desired_correction) else desired_correction

  arrival_share = _interp(
    abs(desired_error),
    *PATH_PROJECTED_ARRIVAL_ERROR_BP,
    0.0,
    1.0,
  )
  return full_correction * arrival_share


def _unwind_target(target: float, measured_curvature: float) -> float:
  """Move a delivered target toward zero without crossing its model direction."""
  corrected = target + _clip(
    _deadzone(target - measured_curvature, PATH_UNWIND_ERROR_DEADZONE),
    (-PATH_UNWIND_LIMIT, PATH_UNWIND_LIMIT),
  )
  return 0.0 if corrected * target < 0.0 else corrected


def _compose_path_target(frame: _PathFrame, allocated_c3: float) -> tuple[PathCoefficients, float, bool]:
  """Resolve model samples and action into one non-duplicated Ford polynomial.

  pathOffset and pathAngle are independent observations of the model trajectory,
  while curvature and curvatureRate are action and slope. Convert the first two
  to curvature observations and resolve the full-polynomial endpoint. update()
  crossfades to this endpoint only when reversible C2 is insufficient.
  """
  desired_curvature = frame.raw_target[2]
  offset_curvature = frame.offset_curvature if frame.valid else desired_curvature
  angle_curvature = frame.angle_curvature if frame.valid else desired_curvature
  geometry_demand = max(abs(offset_curvature), abs(angle_curvature))
  geometry_is_coherent = frame.valid and offset_curvature * angle_curvature > 0.0

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

  stale_model_geometry = (
    geometry_share > 0.0
    and model_target * frame.raw_target[3] < 0.0
    and model_target * frame.desired_curvature < 0.0
    and frame.measured_curvature * model_target > 0.0
  )
  if stale_model_geometry:
    geometry_share = 0.0
    model_target = frame.desired_curvature
    offset_target = frame.desired_curvature
    angle_target = frame.desired_curvature

  coherent_model_maneuver = geometry_share > 0.0
  wheel_beyond_action = _target_is_behind_wheel(desired_curvature, frame.measured_curvature)
  wheel_beyond_model = not coherent_model_maneuver or _target_is_behind_wheel(model_target, frame.measured_curvature)
  wheel_beyond_target = wheel_beyond_action and wheel_beyond_model

  if wheel_beyond_target:
    offset_target = _unwind_target(offset_target, frame.measured_curvature)
    angle_target = _unwind_target(angle_target, frame.measured_curvature)
  else:
    correction_is_coherent = model_target * frame.desired_curvature > 0.0
    model_action_disagreement = model_target * desired_curvature < 0.0 or model_target * frame.measured_curvature < 0.0
    c3_limit = PATH_LIMITS[3][1] if allocated_c3 >= 0.0 else abs(PATH_LIMITS[3][0])
    outward_c3_is_pinned = frame.raw_target[3] * model_target > 0.0 and abs(allocated_c3) >= c3_limit
    continuing_model_preview = (
      correction_is_coherent and outward_c3_is_pinned and abs(model_target) > abs(frame.desired_curvature) + PATH_TRACKING_ERROR_DEADZONE
    )
    bounded_preview_target = math.copysign(
      min(abs(model_target), abs(frame.desired_curvature) + PATH_C0_CONTINUATION_MARGIN),
      model_target,
    )
    c0_tracking_target = bounded_preview_target if continuing_model_preview else frame.desired_curvature
    c0_tracking_limit = (
      PATH_C0_CONTINUATION_ERROR_LIMIT
      if continuing_model_preview
      else (PATH_C1_TRACKING_ERROR_LIMIT if model_action_disagreement else PATH_C0_TRACKING_ERROR_LIMIT)
    )
    offset_target += _gated_tracking_correction(
      offset_target,
      c0_tracking_target if correction_is_coherent else 0.0,
      frame.measured_curvature,
      frame.projected_curvature,
      c0_tracking_limit,
    )
    angle_target += _gated_tracking_correction(
      angle_target,
      frame.desired_curvature if correction_is_coherent else 0.0,
      frame.measured_curvature,
      frame.projected_curvature,
      PATH_C1_TRACKING_ERROR_LIMIT,
    )

  target = (
    0.5 * offset_target * PATH_MIN_LOOKAHEAD**2,
    angle_target * frame.lookahead,
    0.0,
    allocated_c3,
  )
  preserve_model_direction = coherent_model_maneuver
  return target, model_target, preserve_model_direction


def _extend_c0_c1_for_geometry_shortfall(
  coefficients: PathCoefficients,
  frame: _PathFrame,
  residual_share: float,
) -> tuple[float, float, float, float]:
  """Use only unused model C0/C1 to cover a verified steering shortfall."""
  del residual_share  # Remaining raw C0/C1 is the authority bound at every blend share.
  if not frame.valid or frame.lat_ctl_limit != 0:
    return coefficients

  if (
    frame.offset_curvature * frame.angle_curvature <= 0.0
    or frame.offset_curvature * frame.desired_curvature <= 0.0
    or frame.raw_target[2] * frame.desired_curvature < 0.0
    or frame.model_curvature * frame.desired_curvature <= 0.0
  ):
    return coefficients

  tracking_error = frame.projected_tracking_error
  extension_curvature = min(
    abs(_deadzone(tracking_error, PATH_TRACKING_ERROR_DEADZONE)),
    PATH_C1_TRACKING_ERROR_LIMIT,
  )
  geometry_share = _interp(
    min(abs(frame.offset_curvature), abs(frame.angle_curvature)),
    PATH_C2_BASEBAND_BP[1],
    PATH_PREVIEW_BP[1],
    0.0,
    1.0,
  )
  action_share = _interp(
    abs(frame.desired_curvature),
    PATH_C2_BASEBAND_BP[1],
    PATH_PREVIEW_BP[1],
    0.0,
    1.0,
  )
  # Both signals must independently identify a maneuver, but multiplying them
  # twice made the verified shortfall correction disproportionately timid at
  # turn onset. Their geometric mean preserves both zero gates while exposing
  # more of the model's unused C0/C1 as confidence rises.
  extension_curvature *= math.sqrt(geometry_share * action_share)
  if extension_curvature == 0.0:
    return coefficients

  direction = math.copysign(1.0, frame.desired_curvature)
  basis = _basis(PATH_MIN_LOOKAHEAD)
  raw_preview_curvature = sum(basis[i] * frame.raw_target[i] for i in (0, 1))
  command_preview_curvature = sum(basis[i] * coefficients[i] for i in (0, 1))
  aggregate_available_curvature = (raw_preview_curvature - command_preview_curvature) * direction
  if aggregate_available_curvature <= 0.0:
    return coefficients

  return _extend_preview(coefficients, frame.raw_target, direction, extension_curvature, aggregate_available_curvature)


def _extend_c0_for_opposite_side_reversal(
  coefficients: PathCoefficients,
  frame: _PathFrame,
  residual_share: float,
) -> tuple[float, float, float, float]:
  """Bridge verified direction changes with immediately removable C0."""
  if not frame.valid or frame.lat_ctl_limit != 0 or residual_share >= 1.0 or abs(frame.desired_curvature) <= PATH_C2_BASEBAND_BP[0]:
    return coefficients

  # C2 is the steady-state anchor, but the PSCM can retain its preceding arc
  # during a quick reversal. Only bridge while both the measured and projected
  # wheel remain on that old side and the model action and spatial slope agree
  # on the new direction.
  if (
    frame.measured_curvature * frame.desired_curvature >= 0.0
    or frame.projected_curvature * frame.desired_curvature >= 0.0
    or frame.raw_target[2] * frame.desired_curvature <= 0.0
    or frame.raw_target[3] * frame.desired_curvature <= 0.0
  ):
    return coefficients

  spatial_demand = abs(frame.raw_target[3]) * PATH_MIN_LOOKAHEAD / 3.0
  if spatial_demand <= PATH_TRACKING_ERROR_DEADZONE:
    return coefficients

  tracking_error = frame.projected_tracking_error
  correction_curvature = min(
    abs(_deadzone(tracking_error, PATH_TRACKING_ERROR_DEADZONE)),
    PATH_UNWIND_LIMIT,
  )
  if correction_curvature == 0.0:
    return coefficients

  action_share = _interp(
    abs(frame.desired_curvature),
    *PATH_C2_BASEBAND_BP,
    0.0,
    1.0,
  )
  old_side_share = min(
    _interp(abs(frame.measured_curvature), *PATH_MEASURED_ARRIVAL_ERROR_BP, 0.0, 1.0),
    _interp(abs(frame.projected_curvature), *PATH_PROJECTED_ARRIVAL_ERROR_BP, 0.0, 1.0),
  )
  direction = math.copysign(1.0, frame.desired_curvature)
  command_curvature = _equivalent_curvature(coefficients)
  corridor_curvature = abs(frame.desired_curvature) + correction_curvature
  missing_curvature = max(corridor_curvature - command_curvature * direction, 0.0)
  extension_curvature = min(missing_curvature, correction_curvature) * action_share * old_side_share
  if extension_curvature == 0.0:
    return coefficients

  values = list(coefficients)
  c0_basis = _basis(PATH_MIN_LOOKAHEAD)[0]
  values[0] = _clip(
    values[0] + direction * extension_curvature / c0_basis,
    PATH_LIMITS[0],
  )
  return tuple(values)


def _plan_authority(frame: _PathFrame, last_c2_anchor: float, last_allocated_c3: float) -> _AuthorityPlan:
  """Allocate sticky C2/C3 and derive the complete fast-coefficient target."""
  residual_share, proactive_preview_share = _preview_allocation(frame)

  target_c2 = _clip(frame.raw_target[2] * (1.0 - residual_share), PATH_LIMITS[2])
  anchored_c2 = _apply_c2_attack(target_c2, last_c2_anchor)
  c3_share = 1.0
  if frame.raw_target[3] * frame.desired_curvature < 0.0:
    tracking_error = frame.desired_curvature - frame.projected_curvature
    if tracking_error * frame.desired_curvature > 0.0:
      c3_share -= _interp(abs(tracking_error), *PATH_C3_UNWIND_ERROR_BP, 0.0, 1.0)
  safe_c3 = _limit_attack(
    _clip(frame.raw_target[3] * c3_share * residual_share, PATH_LIMITS[3]),
    last_allocated_c3,
    PATH_C3_SLEW,
  )
  full_target, model_curvature, preserve_model_direction = _compose_path_target(frame, safe_c3)
  c2_shortfall = target_c2 - anchored_c2
  preview_curvature = sum(_basis(PATH_MIN_LOOKAHEAD)[i] * full_target[i] for i in (0, 1))
  coherent_handoff = (
    c2_shortfall * preview_curvature > 0.0 and c2_shortfall * frame.desired_curvature > 0.0 and preview_curvature * frame.desired_curvature > 0.0
  )
  if frame.lat_ctl_limit == 0 and coherent_handoff:
    same_side_tracking = frame.measured_curvature * frame.desired_curvature > 0.0 and frame.projected_curvature * frame.desired_curvature > 0.0
    if same_side_tracking and frame.projected_tracking_error != 0.0:
      residual_share = _clip(residual_share + abs(c2_shortfall / preview_curvature), (residual_share, 1.0))
    elif abs(target_c2) >= PATH_PREVIEW_BP[1]:
      base_command = full_target[0] * residual_share, full_target[1] * residual_share, anchored_c2, full_target[3]
      direction = math.copysign(1.0, frame.desired_curvature)
      desired_shortfall = max(abs(frame.desired_curvature) - _equivalent_curvature(base_command) * direction, 0.0)
      missing_share = min(abs(c2_shortfall), desired_shortfall) / abs(preview_curvature)
      residual_share = _clip(residual_share + missing_share, (residual_share, 1.0))
  bounds = list(PATH_LIMITS)
  bounds[2] = (anchored_c2, anchored_c2)
  bounds[3] = (safe_c3, safe_c3)
  return _AuthorityPlan(
    residual_share,
    proactive_preview_share,
    anchored_c2,
    safe_c3,
    full_target,
    model_curvature,
    preserve_model_direction,
    tuple(bounds),
  )


def _solve_fast_path(frame: _PathFrame, plan: _AuthorityPlan) -> PathCoefficients:
  """Solve C0/C1 around the allocated sticky coefficients."""
  target = (
    plan.full_target[0] * plan.residual_share,
    plan.full_target[1] * plan.residual_share,
    plan.anchored_c2,
    plan.full_target[3],
  )
  coefficients = tuple(_clip(value, bound) for value, bound in zip(target, plan.bounds, strict=True))
  coefficients = _extend_c0_c1_for_proactive_preview(
    coefficients,
    frame,
    plan.proactive_preview_share,
  )
  return coefficients


def _apply_output_envelope(frame: _PathFrame, plan: _AuthorityPlan, coefficients: PathCoefficients) -> tuple[PathCoefficients, bool]:
  """Apply the ordered PSCM, reversal, direction, and arrival envelope."""
  unallocated_c3 = coefficients[3]
  envelope_share = 0.5 if frame.lat_ctl_limit == 1 else 1.0 if frame.lat_ctl_limit == 2 else 0.0
  c0, c1, c2, c3 = coefficients
  if envelope_share != 0.0 and c3 * frame.desired_curvature > 0.0 and c3 * plan.model_curvature > 0.0:
    arrival_share = _interp(abs(frame.projected_tracking_error), *PATH_PROJECTED_ARRIVAL_ERROR_BP, 0.0, 1.0)
    requested_spill = c3 * envelope_share * arrival_share
    c0_with_spill = _clip(c0 + requested_spill * PATH_C3_TO_C0_LOOKAHEAD**3 / 6.0, PATH_LIMITS[0])
    actual_spill = (c0_with_spill - c0) * 6.0 / PATH_C3_TO_C0_LOOKAHEAD**3
    coefficients = c0_with_spill, c1, c2, c3 - actual_spill

  coefficients = _extend_c0_c1_for_geometry_shortfall(
    coefficients,
    frame,
    plan.residual_share,
  )
  c3_was_reallocated = coefficients[3] != unallocated_c3
  coefficients = _extend_c0_for_opposite_side_reversal(
    coefficients,
    frame,
    plan.residual_share,
  )

  if plan.preserve_model_direction:
    command_curvature = _equivalent_curvature(coefficients)
    if plan.model_curvature * command_curvature < 0.0:
      guarded_curvature = math.copysign(PATH_DIRECTION_MARGIN, plan.model_curvature)
      values = list(coefficients)
      basis = _basis(PATH_MIN_LOOKAHEAD)
      for i in (0, 1):
        correction = (guarded_curvature - command_curvature) / basis[i]
        values[i] = _clip(values[i] + correction, plan.bounds[i])
        command_curvature = _equivalent_curvature(tuple(values))
        if plan.model_curvature * command_curvature >= 0.0:
          coefficients = tuple(values)
          break
      else:
        coefficients = 0.0, 0.0, 0.0, 0.0

  if frame.measured_curvature != 0.0:
    direction = math.copysign(1.0, frame.measured_curvature)
    desired_outward_curvature = max(frame.desired_curvature * direction, 0.0)
    beyond_error = max(abs(frame.measured_curvature) - desired_outward_curvature, 0.0)
    arrival_share = _interp(beyond_error, *PATH_MEASURED_ARRIVAL_ERROR_BP, 0.0, 1.0)
    release_demand = max(-frame.raw_target[3] * direction * frame.lookahead / 3.0, 0.0)
    spatial_release_share = _interp(release_demand, *PATH_C2_BASEBAND_BP, 0.0, 1.0)
    action_support_ratio = desired_outward_curvature / abs(frame.measured_curvature)
    action_support_share = _interp(action_support_ratio, *PATH_ACTION_SUPPORT_BP, 0.0, 1.0)
    release_share = arrival_share * max(spatial_release_share, 1.0 - action_support_share)
    if release_share != 0.0:
      command_curvature = _equivalent_curvature(coefficients)
      corridor_margin = _blend(PATH_ORDINARY_ARRIVAL_MARGIN, PATH_C0_CONTINUATION_MARGIN, action_support_share)
      excess_curvature = command_curvature * direction - (desired_outward_curvature + corridor_margin)
      basis = _basis(PATH_MIN_LOOKAHEAD)
      outward_preview_curvature = sum(basis[i] * coefficients[i] * direction for i in (0, 1) if coefficients[i] * direction > 0.0)
      if excess_curvature > 0.0 and outward_preview_curvature > 0.0:
        removed_curvature = min(excess_curvature * release_share, outward_preview_curvature)
        preview_share = 1.0 - removed_curvature / outward_preview_curvature
        values = list(coefficients)
        for i in (0, 1):
          if values[i] * direction > 0.0:
            values[i] *= preview_share
        coefficients = tuple(values)
  return coefficients, c3_was_reallocated


class ProjectedLatControlPath:
  """Return one coherent, bounded Ford polynomial through a stable interface."""

  def __init__(self):
    self._last_c2_anchor = 0.0
    self._last_allocated_c3 = 0.0

  def update(
    self,
    path,
    measured_curvature: float,
    v_ego: float,
    active: bool,
    driver_override: bool,
    projected_measured_curvature: float | None = None,
    desired_angle_curvature: float | None = None,
    lat_ctl_limit: int = 0,
  ) -> LateralPathCommand:
    if not active:
      self._last_c2_anchor = 0.0
      self._last_allocated_c3 = 0.0
      return LateralPathCommand()

    frame = _observe_path(
      path,
      measured_curvature,
      projected_measured_curvature,
      desired_angle_curvature,
      v_ego,
      lat_ctl_limit,
    )

    if driver_override:
      command = LateralPathCommand(
        valid=frame.valid,
        path_offset=_clip(0.5 * frame.measured_curvature * PATH_MIN_LOOKAHEAD**2, PATH_LIMITS[0]),
        path_angle=_clip(frame.measured_curvature * max(frame.v_ego, PATH_MIN_LOOKAHEAD), PATH_LIMITS[1]),
      )
      self._last_c2_anchor = 0.0
      self._last_allocated_c3 = 0.0
      return command

    plan = _plan_authority(frame, self._last_c2_anchor, self._last_allocated_c3)
    coefficients = _solve_fast_path(frame, plan)
    coefficients, c3_was_reallocated = _apply_output_envelope(frame, plan, coefficients)
    command = LateralPathCommand(
      valid=frame.valid, path_offset=coefficients[0], path_angle=coefficients[1], curvature=coefficients[2], curvature_rate=coefficients[3]
    )
    self._last_c2_anchor = plan.anchored_c2
    self._last_allocated_c3 = plan.safe_c3 if c3_was_reallocated else command.curvature_rate
    return command
