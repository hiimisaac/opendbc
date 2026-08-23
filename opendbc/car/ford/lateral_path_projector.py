"""Compact preview-path control for Ford LMC2 steering."""

from __future__ import annotations

from dataclasses import dataclass
import math


PATH_LIMITS = (
  (-4.61, 4.60),
  (-0.475, 0.497),
  (-0.02, 0.02),
  (-0.001024, 0.001023),
)
PATH_MIN_PREVIEW_DISTANCE = 7.0
PATH_MAX_PREVIEW_DISTANCE = 12.0
PATH_PREVIEW_TIME = 0.35
PATH_C2_MANEUVER_BP = (0.003, 0.006)
PATH_C2_UNDERTRACKING_BP = (0.006, 0.009)
PATH_C3_MANEUVER_BP = (0.003, 0.006)
PATH_PREVIEW_GEOMETRY_BP = (0.0015, 0.0045)
PATH_TRACKING_ERROR_BP = (0.003, 0.006)
PATH_ARC_GEOMETRY_BP = (0.003, 0.012)
PATH_TRACKING_ERROR_DEADZONE = 0.0005
PATH_C0_TRACKING_ERROR_LIMIT = 0.02
PATH_C1_TRACKING_ERROR_LIMIT = 0.012
PATH_C2_SLEW = 0.0002
PATH_C3_SLEW = 0.0002
PATH_ARRIVAL_BP = (0.0005, 0.002)
PATH_ARC_UNWIND_DEADZONE = 0.0005
PATH_ACTION_SUPPORT_BP = (0.1, 0.2)
PATH_ARRIVAL_MARGIN = (0.00025, 0.006)


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


def _interp(value: float, lower: float, upper: float) -> float:
  if value <= lower:
    return 0.0
  if value >= upper:
    return 1.0
  return (value - lower) / (upper - lower)


def _deadzone(value: float, deadzone: float) -> float:
  return math.copysign(max(abs(value) - deadzone, 0.0), value)


def _limit_attack(value: float, previous: float, max_step: float) -> float:
  """Rate-limit sticky-channel growth while allowing immediate release."""
  if value * previous < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(previous):
    return math.copysign(min(abs(value), abs(previous) + max_step), value)
  return value


def _basis(distance: float) -> tuple[float, float, float, float]:
  return 2.0 / distance ** 2, 2.0 / distance, 1.0, distance / 3.0


def _equivalent_curvature(coefficients: tuple[float, float, float, float],
                          distance: float = PATH_MIN_PREVIEW_DISTANCE) -> float:
  return sum(
    basis * coefficient
    for basis, coefficient in zip(_basis(distance), coefficients, strict=True)
  )


def _preview_distance(v_ego: float) -> float:
  return _clip(
    max(v_ego, 0.0) * PATH_PREVIEW_TIME,
    (PATH_MIN_PREVIEW_DISTANCE, PATH_MAX_PREVIEW_DISTANCE),
  )


def _conservative_tracking_error(target: float, measured: float, projected: float) -> float:
  """Use projected wheel motion only while it approaches without crossing."""
  measured_error = target - measured
  projected_error = target - projected
  projected_motion = projected - measured
  if measured_error == 0.0 or projected_motion * measured_error <= 0.0:
    return measured_error
  if projected_error * measured_error <= 0.0:
    return 0.0
  return projected_error if abs(projected_error) < abs(measured_error) else measured_error


def _maneuver_demand(raw: tuple[float, float, float, float],
                     v_ego: float, valid: bool) -> float:
  """Return model demand that reversible C2 cannot carry cleanly."""
  lookahead = max(v_ego, PATH_MIN_PREVIEW_DISTANCE)
  c2_overflow = max(abs(raw[2]) - PATH_LIMITS[2][1], 0.0)
  spatial_change = abs(raw[3]) * lookahead / 3.0
  if not valid:
    return max(c2_overflow, spatial_change)

  offset_curvature = 2.0 * raw[0] / PATH_MIN_PREVIEW_DISTANCE ** 2
  angle_curvature = raw[1] / lookahead
  coherent_geometry = min(abs(offset_curvature), abs(angle_curvature)) \
    if offset_curvature * angle_curvature > 0.0 else 0.0
  return max(c2_overflow, spatial_change, coherent_geometry - PATH_LIMITS[2][1], 0.0)


def _model_arc_target(raw: tuple[float, float, float, float],
                      v_ego: float, valid: bool) -> float:
  """Resolve model action and coherent path pose into one future arc."""
  if not valid:
    return raw[2]

  lookahead = max(v_ego, PATH_MIN_PREVIEW_DISTANCE)
  offset_curvature = 2.0 * raw[0] / PATH_MIN_PREVIEW_DISTANCE ** 2
  angle_curvature = raw[1] / lookahead
  if offset_curvature * angle_curvature <= 0.0:
    return raw[2]

  geometry_curvature = math.copysign(
    min(abs(offset_curvature), abs(angle_curvature)),
    offset_curvature,
  )
  geometry_share = _interp(
    max(abs(offset_curvature), abs(angle_curvature)),
    *PATH_ARC_GEOMETRY_BP,
  )
  reference = raw[2] if geometry_curvature * raw[2] >= 0.0 else 0.0
  return reference + geometry_share * (geometry_curvature - reference)


def _bridge_c2_handoff(arc_share: float, arc_target: float,
                       target_c2: float, anchored_c2: float,
                       desired: float | None) -> float:
  """Keep the fast arc until the slow C2 command has assumed its share."""
  if desired is None or arc_target == 0.0:
    return arc_share

  c2_shortfall = target_c2 - anchored_c2
  if c2_shortfall * arc_target <= 0.0 or desired * arc_target <= 0.0:
    return arc_share

  desired_room = max(abs(desired) - abs(anchored_c2), 0.0)
  bridge = min(abs(c2_shortfall), desired_room)
  return max(arc_share, min(bridge / abs(arc_target), 1.0))


def _c3_compatibility_share(curvature_rate: float, desired: float | None,
                            projected: float) -> float:
  """Delay an opposing spatial slope until the wheel reaches desired."""
  if desired is None or curvature_rate * desired >= 0.0:
    return 1.0
  tracking_error = desired - projected
  if tracking_error * desired <= 0.0:
    return 1.0
  return 1.0 - _interp(abs(tracking_error), *PATH_ARRIVAL_BP)


def _bound_nonfast_anchor(target: float, c2: float, c3: float,
                          distance: float) -> tuple[float, float]:
  """Prevent sticky C2/C3 from overrunning the model's active anchor."""
  if abs(target) < 1e-12:
    return c2, c3

  direction = math.copysign(1.0, target)
  excess = direction * (c2 + c3 * distance / 3.0) - abs(target)
  if excess <= 0.0:
    return c2, c3

  c3_reduction = min(excess, max(direction * c3 * distance / 3.0, 0.0))
  c3 -= direction * c3_reduction * 3.0 / distance
  excess -= c3_reduction
  c2 -= direction * min(excess, max(direction * c2, 0.0))
  return c2, c3


def _encode_fast_arc(offset_curvature: float, angle_curvature: float,
                     lookahead: float) -> tuple[float, float]:
  """Describe one future arc through Ford's fast path-pose channels."""
  return (
    0.5 * offset_curvature * PATH_MIN_PREVIEW_DISTANCE ** 2,
    angle_curvature * max(lookahead, PATH_MIN_PREVIEW_DISTANCE),
  )


def _fast_tracking_correction(model_target: float, desired: float | None,
                              measured: float, projected: float,
                              ownership_share: float) -> tuple[float, float]:
  """Return immediately removable C0/C1 curvature while steering is behind."""
  if desired is None or model_target * desired <= 0.0 or ownership_share <= 0.0:
    return 0.0, 0.0

  desired_error = _conservative_tracking_error(desired, measured, projected)
  if desired_error * desired <= 0.0:
    return 0.0, 0.0

  model_error = _conservative_tracking_error(model_target, measured, projected)
  error = desired_error
  if model_error * desired_error > 0.0 and abs(model_error) > abs(error):
    error = model_error

  correction = _deadzone(error, PATH_TRACKING_ERROR_DEADZONE)
  correction *= ownership_share * _interp(abs(desired_error), *PATH_ARRIVAL_BP)
  return (
    _clip(correction, (-PATH_C0_TRACKING_ERROR_LIMIT, PATH_C0_TRACKING_ERROR_LIMIT)),
    _clip(correction, (-PATH_C1_TRACKING_ERROR_LIMIT, PATH_C1_TRACKING_ERROR_LIMIT)),
  )


def _taper_arrived_arc(target: float, desired: float,
                       measured: float, projected: float) -> float:
  """Trim an outward arc only after both wheel estimates pass its corridor."""
  if target == 0.0 or desired * target <= 0.0:
    return target

  direction = math.copysign(1.0, target)
  delivered = min(measured * direction, projected * direction)
  corridor = max(target * direction, desired * direction)
  excess = delivered - corridor - PATH_ARC_UNWIND_DEADZONE
  if excess <= 0.0:
    return target
  return direction * max(target * direction - excess, 0.0)


def _bound_arrived_arc(coefficients: tuple[float, float, float, float],
                       model_curvature_rate: float, desired: float | None,
                       measured: float, projected: float,
                       v_ego: float) -> tuple[float, float, float, float]:
  """Keep an inward-sloping path inside the arrived steering corridor."""
  if desired is None or desired == 0.0:
    return coefficients

  direction = math.copysign(1.0, desired)
  delivered = min(measured * direction, projected * direction)
  desired_magnitude = desired * direction
  if delivered <= 0.0 or model_curvature_rate * direction >= 0.0:
    return coefficients

  arrival_share = 1.0 - _interp(
    max(desired_magnitude - delivered, 0.0),
    *PATH_ARRIVAL_BP,
  )
  inward_slope = -model_curvature_rate * direction * \
                 max(v_ego, PATH_MIN_PREVIEW_DISTANCE) / 3.0
  release_share = arrival_share * _interp(inward_slope, *PATH_C3_MANEUVER_BP)
  if release_share <= 0.0:
    return coefficients

  action_support = _interp(
    desired_magnitude / max(delivered, 1e-9),
    *PATH_ACTION_SUPPORT_BP,
  )
  margin = PATH_ARRIVAL_MARGIN[0] + \
           action_support * (PATH_ARRIVAL_MARGIN[1] - PATH_ARRIVAL_MARGIN[0])
  excess = max(_equivalent_curvature(coefficients) * direction - desired_magnitude - margin, 0.0)
  excess *= release_share
  if excess <= 0.0:
    return coefficients

  basis = _basis(PATH_MIN_PREVIEW_DISTANCE)
  outward_fast = sum(
    max(coefficients[index] * basis[index] * direction, 0.0)
    for index in (0, 1)
  )
  if outward_fast <= 0.0:
    return coefficients

  scale = max(1.0 - min(excess, outward_fast) / outward_fast, 0.0)
  values = list(coefficients)
  for index in (0, 1):
    if values[index] * direction > 0.0:
      values[index] *= scale
  return tuple(values)


def lmc2_control_utilization(command: LateralPathCommand, lat_ctl_limit: int) -> float:
  """Return signed coefficient-envelope usage augmented by PSCM limit status."""
  coefficients = command.coefficients()
  utilization = [
    abs(value) / (limits[1] if value >= 0.0 else abs(limits[0]))
    for value, limits in zip(coefficients, PATH_LIMITS, strict=True)
  ]
  magnitude = _clip(max(utilization), (0.0, 1.0))
  if magnitude == 0.0:
    return 0.0
  if lat_ctl_limit == 1:
    magnitude = max(magnitude, 0.8)
  elif lat_ctl_limit == 2:
    magnitude = 1.0

  direction_source = _equivalent_curvature(coefficients)
  if abs(direction_source) < 1e-9:
    direction_source = coefficients[max(range(4), key=utilization.__getitem__)]
  return math.copysign(magnitude, direction_source)


class ProjectedLatControlPath:
  """Convert one model path and measured wheel motion into one LMC2 command."""

  def __init__(self):
    self._last_c2_anchor: float | None = None
    self._last_c3 = 0.0

  def _reset(self) -> LateralPathCommand:
    self._last_c2_anchor = None
    self._last_c3 = 0.0
    return LateralPathCommand()

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    measured = _finite(measured_curvature)
    projected = measured if projected_measured_curvature is None else \
                _finite(projected_measured_curvature, measured)
    desired = None if desired_angle_curvature is None else _finite(desired_angle_curvature)
    v_ego = max(_finite(v_ego), 0.0)
    if not active:
      return self._reset()

    valid = path is not None and bool(getattr(path, "valid", False))
    raw = (
      _finite(getattr(path, "pathOffset", 0.0)) if valid else 0.0,
      _finite(getattr(path, "pathAngle", 0.0)) if valid else 0.0,
      _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0,
      _finite(getattr(path, "curvatureRate", 0.0)) if valid else 0.0,
    )
    distance = _preview_distance(v_ego)

    if driver_override:
      self._last_c2_anchor = None
      self._last_c3 = 0.0
      return LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured * distance ** 2, PATH_LIMITS[0]),
        path_angle=_clip(measured * distance, PATH_LIMITS[1]),
      )

    if not valid:
      return LateralPathCommand(curvature=_clip(raw[2], PATH_LIMITS[2]))

    model_target = _model_arc_target(raw, v_ego, valid)
    target_equivalent = _equivalent_curvature(raw, distance)
    preview_equivalent = target_equivalent - raw[2]
    tracking_error = _conservative_tracking_error(target_equivalent, measured, projected)

    maneuver_share = _interp(_maneuver_demand(raw, v_ego, valid), *PATH_C2_MANEUVER_BP)
    preview_share = _interp(abs(preview_equivalent), *PATH_PREVIEW_GEOMETRY_BP)
    preview_release = preview_share * _interp(abs(tracking_error), *PATH_TRACKING_ERROR_BP) \
      if preview_equivalent * tracking_error > 0.0 else 0.0
    undertracking_share = 0.0
    if tracking_error * target_equivalent > 0.0:
      undertracking_share = min(
        _interp(abs(raw[2]), *PATH_C2_UNDERTRACKING_BP),
        _interp(abs(tracking_error), *PATH_TRACKING_ERROR_BP),
      )
    ownership_share = max(maneuver_share, preview_release, undertracking_share)

    target_c2 = _clip(raw[2] * (1.0 - ownership_share), PATH_LIMITS[2])
    c2 = target_c2 if self._last_c2_anchor is None else \
         _limit_attack(target_c2, self._last_c2_anchor, PATH_C2_SLEW)
    ownership_share = _bridge_c2_handoff(
      ownership_share, model_target, target_c2, c2, desired,
    )

    c3_share = _c3_compatibility_share(raw[3], desired, projected)
    target_c3 = _clip(raw[3] * maneuver_share * c3_share, PATH_LIMITS[3])
    c3 = _limit_attack(target_c3, self._last_c3, PATH_C3_SLEW)
    c2, c3 = _bound_nonfast_anchor(target_equivalent, c2, c3, distance)

    arc_target = model_target * ownership_share
    if desired is not None:
      arc_target = _taper_arrived_arc(arc_target, desired, measured, projected)
    c0_correction, c1_correction = _fast_tracking_correction(
      model_target, desired, measured, projected, ownership_share,
    )
    c0, c1 = _encode_fast_arc(
      arc_target + c0_correction,
      arc_target + c1_correction,
      distance,
    )

    coefficients = tuple(
      _clip(value, limits)
      for value, limits in zip((c0, c1, c2, c3), PATH_LIMITS, strict=True)
    )
    coefficients = _bound_arrived_arc(
      coefficients, raw[3], desired, measured, projected, v_ego,
    )
    self._last_c2_anchor = c2
    self._last_c3 = c3
    return LateralPathCommand(
      valid=valid,
      path_offset=coefficients[0],
      path_angle=coefficients[1],
      curvature=coefficients[2],
      curvature_rate=coefficients[3],
    )
