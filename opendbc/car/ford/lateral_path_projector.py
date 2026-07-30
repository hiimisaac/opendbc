"""Stateless Ford LMC2 polynomial controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


Coefficients = tuple[float, float, float, float]
Bounds = tuple[tuple[float, float], ...]

PATH_LIMITS: Bounds = (
  (-4.61, 4.60),
  (-0.475, 0.497),
  (-0.02, 0.02),
  (-0.001024, 0.001023),
)
NEAR_DISTANCE = 7.0
NEAR_BASIS = (2.0 / NEAR_DISTANCE ** 2, 2.0 / NEAR_DISTANCE, 1.0, NEAR_DISTANCE / 3.0)
MANEUVER_BP = (0.003, 0.006)
PREVIEW_BP = (0.003, 0.012)
ARRIVAL_BP = (0.0005, 0.002)

TRACKING_DEADZONE = 0.0005
TRACKING_EXTENSION_DEADZONE = 0.00025
TRACKING_EXTENSION_LIMIT = 0.004
FEEDBACK_PREVIEW_LIMIT = 0.0015
CLIPPED_RECOVERY_LIMIT = 0.006
C0_TRACKING_LIMIT = 0.02
C0_CONTINUATION_LIMIT = 0.04
C1_TRACKING_LIMIT = 0.012
CONTINUATION_MARGIN = 0.006
UNWIND_LIMIT = 0.006
DIRECTION_MARGIN = 0.0005
SPATIAL_ONSET_DISTANCE = 12.0
SPATIAL_ONSET_RELATIVE_MIN = 0.5
C3_REALLOCATION_DISTANCE = 15.0


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0

  def coefficients(self) -> Coefficients:
    return self.path_offset, self.path_angle, self.curvature, self.curvature_rate


class LateralPathController(Protocol):
  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    ...


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _ramp(value: float, breakpoints: tuple[float, float]) -> float:
  lower, upper = breakpoints
  if value <= lower:
    return 0.0
  if value >= upper:
    return 1.0
  return (value - lower) / (upper - lower)


def _deadzone(value: float, deadzone: float) -> float:
  return math.copysign(max(abs(value) - deadzone, 0.0), value)


def _lerp(first: float, second: float, second_share: float) -> float:
  return first + second_share * (second - first)


def _curvature(coefficients: Coefficients) -> float:
  return sum(NEAR_BASIS[i] * coefficients[i] for i in range(4))


def _shortfall(target: float, measured: float, projected: float) -> float:
  """Smallest outward error while both wheel estimates remain behind target."""
  measured_error = target - measured
  projected_error = target - projected
  if measured_error * target <= 0.0 or projected_error * target <= 0.0:
    return 0.0
  return math.copysign(min(abs(measured_error), abs(projected_error)), target)


def _arrival_share(target: float, measured: float, projected: float) -> float:
  return _ramp(abs(_shortfall(target, measured, projected)), ARRIVAL_BP)


def _add_c0(coefficients: Coefficients, curvature: float) -> Coefficients:
  values = list(coefficients)
  values[0] = _clip(values[0] + curvature / NEAR_BASIS[0], PATH_LIMITS[0])
  return tuple(values)


def _outward_extension(command: float, desired: float, measured: float,
                       projected: float, limit: float) -> float:
  wheel_error = _shortfall(desired, measured, projected)
  command_error = desired - command
  if wheel_error * desired <= 0.0 or command_error * desired <= 0.0:
    return 0.0
  available_error = min(abs(wheel_error), abs(command_error))
  extension = min(
    max(available_error - TRACKING_EXTENSION_DEADZONE, 0.0),
    limit,
  )
  return math.copysign(extension, desired)


def _retain_feedback_preview(coefficients: Coefficients, raw: Coefficients,
                             desired: float, measured: float,
                             projected: float, retention_share: float) -> Coefficients:
  """Keep bounded coherent C0/C1 preview until both wheel estimates arrive."""
  shortfall = _shortfall(desired, measured, projected)
  raw_c0_curvature = raw[0] * NEAR_BASIS[0]
  raw_c1_curvature = raw[1] * NEAR_BASIS[1]
  local_curvature = raw[2] + raw[3] * NEAR_DISTANCE
  wheel_has_moved = max(abs(measured), abs(projected)) > TRACKING_DEADZONE
  direction = math.copysign(1.0, desired)
  local_support = _ramp(local_curvature * direction, (0.0, 0.5 * abs(desired)))
  maneuver_support = _ramp(abs(desired), MANEUVER_BP) * local_support * retention_share
  if shortfall == 0.0 or maneuver_support == 0.0 or not wheel_has_moved or \
      _curvature(coefficients) * desired <= 0.0 or \
      raw_c0_curvature * desired <= 0.0 or \
      raw_c1_curvature * desired <= 0.0:
    return coefficients

  available_preview = direction * (raw_c0_curvature + raw_c1_curvature)
  target_extension = min(
    max(abs(shortfall) - TRACKING_EXTENSION_DEADZONE, 0.0),
    available_preview,
    FEEDBACK_PREVIEW_LIMIT,
  )
  current_extension = direction * (_curvature(coefficients) - desired)
  correction = min(
    max(target_extension - current_extension, 0.0),
    FEEDBACK_PREVIEW_LIMIT,
  ) * maneuver_support
  if correction == 0.0:
    return coefficients

  c0_share = direction * raw_c0_curvature / available_preview
  c1_share = direction * raw_c1_curvature / available_preview
  values = list(coefficients)
  values[0] = _clip(
    values[0] + direction * correction * c0_share / NEAR_BASIS[0],
    PATH_LIMITS[0],
  )
  values[1] = _clip(
    values[1] + direction * correction * c1_share / NEAR_BASIS[1],
    PATH_LIMITS[1],
  )
  return tuple(values)


def _tracking_correction(model_target: float, desired_target: float,
                         measured: float, projected: float, limit: float) -> float:
  desired_error = _shortfall(desired_target, measured, projected)
  desired_correction = _clip(
    _deadzone(desired_error, TRACKING_DEADZONE),
    (-limit, limit),
  )
  if desired_correction == 0.0:
    return 0.0

  model_error = model_target - measured
  model_correction = 0.0
  if model_error * model_target > 0.0:
    model_correction = _clip(
      _deadzone(model_error, TRACKING_DEADZONE),
      (-limit, limit),
    )
  if model_correction * desired_correction <= 0.0:
    correction = desired_correction
  else:
    correction = model_correction if abs(model_correction) >= abs(desired_correction) else desired_correction
  return correction * _ramp(abs(desired_error), ARRIVAL_BP)


def _unwind(target: float, measured: float) -> float:
  corrected = target + _clip(
    _deadzone(target - measured, TRACKING_DEADZONE),
    (-UNWIND_LIMIT, UNWIND_LIMIT),
  )
  return 0.0 if corrected * target < 0.0 else corrected


def _compose(raw: Coefficients, valid: bool, measured: float, projected: float,
             desired: float, speed: float) -> tuple[Coefficients, float, float, bool, float]:
  """Normalize model geometry and compose one requested spatial polynomial."""
  c0, c1, c2, c3 = raw
  lookahead = max(speed, NEAR_DISTANCE)
  offset_curvature = 2.0 * c0 / NEAR_DISTANCE ** 2 if valid else c2
  angle_curvature = c1 / lookahead if valid else c2
  geometry_coherent = valid and offset_curvature * angle_curvature > 0.0

  rate_demand = abs(c3) * lookahead / 3.0
  geometry_demand = min(abs(offset_curvature), abs(angle_curvature)) \
    if geometry_coherent else 0.0
  maneuver_demand = max(
    rate_demand,
    abs(c2) - PATH_LIMITS[2][1],
    geometry_demand - PATH_LIMITS[2][1],
    0.0,
  )

  spatial_change = abs(c3) * SPATIAL_ONSET_DISTANCE / 3.0
  desired_error = _shortfall(desired, measured, projected)
  confirmed_onset = valid and desired_error * desired > 0.0 and c3 * desired > 0.0 and \
                    spatial_change >= max(MANEUVER_BP[0], SPATIAL_ONSET_RELATIVE_MIN * abs(desired))
  if confirmed_onset:
    maneuver_demand = max(
      maneuver_demand,
      spatial_change * _ramp(abs(desired_error), ARRIVAL_BP),
    )

  maneuver_share = _ramp(maneuver_demand, MANEUVER_BP)
  safe_c2 = _clip(c2 * (1.0 - maneuver_share), PATH_LIMITS[2])

  c3_share = 1.0
  if c3 * desired < 0.0:
    projected_error = desired - projected
    if projected_error * desired > 0.0:
      c3_share = 1.0 - _ramp(abs(projected_error), ARRIVAL_BP)
  requested_c3 = c3 * c3_share * maneuver_share
  safe_c3 = _clip(requested_c3, PATH_LIMITS[3])

  if geometry_coherent:
    geometry_share = _ramp(max(abs(offset_curvature), abs(angle_curvature)), PREVIEW_BP)
    geometry_curvature = math.copysign(
      min(abs(offset_curvature), abs(angle_curvature)),
      offset_curvature,
    )
    reference = c2 if geometry_curvature * c2 >= 0.0 else 0.0
    model_target = _lerp(reference, geometry_curvature, geometry_share)
    offset_target = _lerp(reference, offset_curvature, geometry_share)
    angle_target = _lerp(reference, angle_curvature, geometry_share)
  else:
    geometry_share = 0.0
    model_target = offset_target = angle_target = c2

  stale_geometry = geometry_share > 0.0 and model_target * c3 < 0.0 and \
                   model_target * desired < 0.0 and measured * model_target > 0.0
  if stale_geometry:
    geometry_share = 0.0
    model_target = offset_target = angle_target = desired

  preserve_direction = geometry_share > 0.0
  leaving_action = c2 * measured <= 0.0 or abs(c2) + TRACKING_DEADZONE < abs(measured)
  leaving_model = not preserve_direction or model_target * measured <= 0.0 or \
                  abs(model_target) + TRACKING_DEADZONE < abs(measured)
  if leaving_action and leaving_model:
    offset_target = _unwind(offset_target, measured)
    angle_target = _unwind(angle_target, measured)
  else:
    correction_coherent = model_target * desired > 0.0
    action_disagrees = model_target * c2 < 0.0 or model_target * measured < 0.0
    c3_limit = PATH_LIMITS[3][1] if safe_c3 >= 0.0 else abs(PATH_LIMITS[3][0])
    outward_c3_pinned = c3 * model_target > 0.0 and abs(safe_c3) >= c3_limit
    continuing_preview = correction_coherent and outward_c3_pinned and \
                         abs(model_target) > abs(desired) + TRACKING_DEADZONE
    bounded_preview = math.copysign(
      min(abs(model_target), abs(desired) + CONTINUATION_MARGIN),
      model_target,
    )
    c0_target = bounded_preview if continuing_preview else desired
    c0_limit = C0_CONTINUATION_LIMIT if continuing_preview else \
               (C1_TRACKING_LIMIT if action_disagrees else C0_TRACKING_LIMIT)
    offset_target += _tracking_correction(
      offset_target,
      c0_target if correction_coherent else 0.0,
      measured,
      projected,
      c0_limit,
    )
    angle_target += _tracking_correction(
      angle_target,
      desired if correction_coherent else 0.0,
      measured,
      projected,
      C1_TRACKING_LIMIT,
    )

  requested = (
    0.5 * offset_target * NEAR_DISTANCE ** 2 * maneuver_share,
    angle_target * lookahead * maneuver_share,
    safe_c2,
    safe_c3,
  )
  if model_target * desired > 0.0:
    requested = _add_c0(
      requested,
      _outward_extension(
        _curvature(requested),
        desired,
        measured,
        projected,
        TRACKING_EXTENSION_LIMIT,
      ),
    )
  return requested, requested_c3, model_target, preserve_direction, maneuver_share


def _project(requested: Coefficients, requested_c3: float, model_target: float,
             preserve_direction: bool, raw_c2: float, raw_c3: float, desired: float,
             measured: float, projected: float, lat_ctl_limit: int) -> Coefficients:
  """Apply the four non-commuting Ford/PSCM constraints in one place."""
  bounds: Bounds = (
    PATH_LIMITS[0],
    PATH_LIMITS[1],
    (requested[2], requested[2]),
    (requested[3], requested[3]),
  )
  unclipped = requested[:3] + (requested_c3,)
  output = tuple(
    _clip(value, bound)
    for value, bound in zip(requested, bounds, strict=True)
  )

  # 1. Recover useful near-field authority lost to signal clipping.
  missing_curvature = _curvature(unclipped) - _curvature(output)
  if missing_curvature * desired > 0.0:
    extension = _outward_extension(
      _curvature(output),
      desired,
      measured,
      projected,
      CLIPPED_RECOVERY_LIMIT,
    )
    if extension * missing_curvature > 0.0:
      recovered = math.copysign(
        min(abs(extension), abs(missing_curvature)),
        missing_curvature,
      )
      output = _add_c0(output, recovered)

  # 2. Inside the PSCM envelope, move outward C3 nearer without changing its
  # 15 m endpoint. Driver-limit and unknown statuses never add authority.
  envelope_share = 0.5 if lat_ctl_limit == 1 else 1.0 if lat_ctl_limit == 2 else 0.0
  c0, c1, c2, c3 = output
  if envelope_share > 0.0 and c3 * desired > 0.0 and c3 * model_target > 0.0:
    requested_spill = c3 * envelope_share * _arrival_share(desired, measured, projected)
    c0_with_spill = _clip(
      c0 + requested_spill * C3_REALLOCATION_DISTANCE ** 3 / 6.0,
      PATH_LIMITS[0],
    )
    actual_spill = (c0_with_spill - c0) * 6.0 / C3_REALLOCATION_DISTANCE ** 3
    output = c0_with_spill, c1, c2, c3 - actual_spill

  # 3. Blend weak opposing preview toward immediate action while both wheel
  # estimates remain behind. Local C2+C3*d support and opposing-command
  # strength crossfade ownership continuously.
  command_curvature = _curvature(output)
  action_share = 0.0
  action_demand_share = _ramp(abs(desired), MANEUVER_BP)
  if lat_ctl_limit == 0 and preserve_direction and model_target * desired < 0.0 and \
      raw_c2 * desired > 0.0 and action_demand_share > 0.0 and abs(desired) <= PATH_LIMITS[2][1] and \
      _shortfall(desired, measured, projected) * desired > 0.0:
    direction = math.copysign(1.0, desired)
    local_action = (raw_c2 + raw_c3 * NEAR_DISTANCE) * direction
    local_share = _ramp(local_action, (0.0, 0.5 * abs(desired)))
    opposing_command = max(-command_curvature * direction, 0.0)
    action_strength_share = 1.0 - _ramp(opposing_command, (abs(desired), 2.0 * abs(desired)))
    action_share = action_demand_share * local_share * action_strength_share
  if preserve_direction and model_target * command_curvature < 0.0:
    guarded_curvature = math.copysign(DIRECTION_MARGIN, model_target)
    values = list(output)
    for index in (0, 1):
      correction = (guarded_curvature - command_curvature) / NEAR_BASIS[index]
      values[index] = _clip(values[index] + correction, bounds[index])
      command_curvature = _curvature(tuple(values))
      if model_target * command_curvature >= 0.0:
        output = tuple(values)
        break
    else:
      output = (0.0, 0.0, 0.0, 0.0)
  if action_share > 0.0:
    command_curvature = _curvature(output)
    action_target = _lerp(command_curvature, desired, action_share)
    output = _add_c0(output, action_target - command_curvature)

  # 4. Once both wheel estimates pass desired angle, constrain only pinned
  # outward preview. Moving desired outward immediately restores full command.
  command_curvature = _curvature(output)
  c3_limit = PATH_LIMITS[3][1] if raw_c3 >= 0.0 else abs(PATH_LIMITS[3][0])
  outward_pinned = raw_c3 * measured > 0.0 and abs(raw_c3) >= c3_limit and \
                   command_curvature * measured > 0.0
  if outward_pinned:
    measured_error = desired - measured
    projected_error = desired - projected
    beyond_error = min(
      abs(measured_error) if measured_error * measured < 0.0 else 0.0,
      abs(projected_error) if projected_error * projected < 0.0 else 0.0,
    )
    arrival_share = _ramp(beyond_error, ARRIVAL_BP)
    if arrival_share > 0.0:
      corridor = abs(desired) + CONTINUATION_MARGIN
      guarded = _lerp(
        abs(command_curvature),
        min(abs(command_curvature), corridor),
        arrival_share,
      )
      scale = guarded / abs(command_curvature)
      output = tuple(value * scale for value in output)
  return output


class ProjectedLatControlPath:
  """Convert model intent and wheel feedback into one bounded Ford path."""

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    measured = _finite(measured_curvature)
    projected = measured if projected_measured_curvature is None else \
                _finite(projected_measured_curvature, measured)
    speed = max(_finite(v_ego), 0.0)
    if not active:
      return LateralPathCommand()

    valid = path is not None and bool(getattr(path, "valid", False))
    if valid:
      raw = (
        _finite(getattr(path, "pathOffset", 0.0)),
        _finite(getattr(path, "pathAngle", 0.0)),
        _finite(getattr(path, "curvature", 0.0)),
        _finite(getattr(path, "curvatureRate", 0.0)),
      )
    else:
      raw = (0.0, 0.0, _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0, 0.0)
    desired = raw[2] if desired_angle_curvature is None else _finite(desired_angle_curvature, raw[2])

    if driver_override:
      lookahead = max(speed, NEAR_DISTANCE)
      return LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured * NEAR_DISTANCE ** 2, PATH_LIMITS[0]),
        path_angle=_clip(measured * lookahead, PATH_LIMITS[1]),
      )

    requested, requested_c3, model_target, preserve_direction, maneuver_share = _compose(
      raw,
      valid,
      measured,
      projected,
      desired,
      speed,
    )
    coefficients = _project(
      requested,
      requested_c3,
      model_target,
      preserve_direction,
      raw[2],
      raw[3],
      desired,
      measured,
      projected,
      lat_ctl_limit,
    )
    if lat_ctl_limit == 0 and maneuver_share < 1.0:
      coefficients = _retain_feedback_preview(
        coefficients,
        raw,
        desired,
        measured,
        projected,
        1.0 - maneuver_share,
      )
    return LateralPathCommand(
      valid=valid,
      path_offset=coefficients[0],
      path_angle=coefficients[1],
      curvature=coefficients[2],
      curvature_rate=coefficients[3],
    )
