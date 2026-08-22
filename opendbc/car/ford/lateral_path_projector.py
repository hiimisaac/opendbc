"""Coherent preview-path control for Ford LMC2 steering."""

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
PATH_C2_BASEBAND = 0.006
PATH_C2_SLEW = 0.0002
PATH_C3_SLEW = 0.0002


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


def _limit_attack(value: float, last: float, max_step: float) -> float:
  """Limit authority growth while allowing an immediate reduction."""
  if value * last < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _apply_c2_attack(value: float, last: float) -> float:
  """Keep ordinary curvature immediate and bound only large C2 transitions."""
  if value * last >= 0.0 and abs(value) <= PATH_C2_BASEBAND:
    return value
  return _limit_attack(value, last, PATH_C2_SLEW)


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


def _path_pose(coefficients: tuple[float, float, float, float],
               distance: float) -> tuple[float, float]:
  c0, c1, c2, c3 = coefficients
  offset = c0 + c1 * distance + 0.5 * c2 * distance ** 2 + c3 * distance ** 3 / 6.0
  angle = c1 + c2 * distance + 0.5 * c3 * distance ** 2
  return offset, angle


def _vehicle_pose(curvature: float, distance: float) -> tuple[float, float]:
  """Project the delivered wheel arc over the same spatial preview."""
  return 0.5 * curvature * distance ** 2, curvature * distance


def _constrain_outward_growth(coefficients: tuple[float, float, float, float],
                              previous: tuple[float, float, float, float],
                              lat_ctl_limit: int) -> tuple[float, float, float, float]:
  """Use PSCM limit feedback as a constraint without freezing path release."""
  if lat_ctl_limit not in (1, 2):  # LimitClose, LimitReached
    return coefficients

  command_curvature = _equivalent_curvature(coefficients)
  previous_curvature = _equivalent_curvature(previous)
  if command_curvature * previous_curvature <= 0.0 or \
     abs(command_curvature) <= abs(previous_curvature):
    return coefficients

  values = list(coefficients)
  basis = _basis(PATH_MIN_PREVIEW_DISTANCE)
  target_curvature = previous_curvature
  for index in (0, 1):
    correction = (target_curvature - _equivalent_curvature(tuple(values))) / basis[index]
    values[index] = _clip(values[index] + correction, PATH_LIMITS[index])
    if abs(_equivalent_curvature(tuple(values))) <= abs(target_curvature) + 1e-9:
      return tuple(values)

  # C0/C1 normally have ample range. If they cannot preserve the envelope,
  # retain the last accepted command rather than extending at a reported limit.
  return previous


def lmc2_control_utilization(command: LateralPathCommand, lat_ctl_limit: int) -> float:
  """Return signed coefficient-envelope usage augmented by PSCM limit status."""
  coefficients = command.coefficients()
  coefficient_utilization = [
    abs(value) / (limits[1] if value >= 0.0 else abs(limits[0]))
    for value, limits in zip(coefficients, PATH_LIMITS, strict=True)
  ]
  coefficient_magnitude = _clip(max(coefficient_utilization), (0.0, 1.0))
  if coefficient_magnitude == 0.0:
    return 0.0

  magnitude = coefficient_magnitude
  if lat_ctl_limit == 1:
    magnitude = max(magnitude, 0.8)
  elif lat_ctl_limit == 2:
    magnitude = 1.0

  direction_source = _equivalent_curvature(coefficients)
  if abs(direction_source) < 1e-9:
    direction_source = coefficients[max(range(4), key=coefficient_utilization.__getitem__)]
  return math.copysign(magnitude, direction_source)


class ProjectedLatControlPath:
  """Convert one cubic path and measured wheel motion into one LMC2 command."""

  def __init__(self):
    self._last_command = LateralPathCommand()
    self._last_c2 = 0.0
    self._last_c3 = 0.0

  def _reset(self) -> LateralPathCommand:
    self._last_command = LateralPathCommand()
    self._last_c2 = 0.0
    self._last_c3 = 0.0
    return self._last_command

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    measured_curvature = _finite(measured_curvature)
    projected_curvature = measured_curvature if projected_measured_curvature is None else \
                          _finite(projected_measured_curvature, measured_curvature)
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

    if driver_override:
      curvature = _clip(measured_curvature, PATH_LIMITS[2])
      command = LateralPathCommand(valid=valid, curvature=curvature)
      self._last_command = command
      self._last_c2 = curvature
      self._last_c3 = 0.0
      return command

    target_c2 = _clip(raw[2], PATH_LIMITS[2])
    anchored_c2 = _apply_c2_attack(target_c2, self._last_c2)
    target_c3 = _clip(raw[3], PATH_LIMITS[3])
    allocated_c3 = _limit_attack(target_c3, self._last_c3, PATH_C3_SLEW)

    if valid:
      distance = _preview_distance(v_ego)
      desired_offset, desired_angle = _path_pose(raw, distance)
      vehicle_offset, vehicle_angle = _vehicle_pose(projected_curvature, distance)
      coefficients = (
        _clip(desired_offset - vehicle_offset, PATH_LIMITS[0]),
        _clip(desired_angle - vehicle_angle, PATH_LIMITS[1]),
        anchored_c2,
        allocated_c3,
      )
    else:
      coefficients = (0.0, 0.0, anchored_c2, 0.0)

    coefficients = _constrain_outward_growth(
      coefficients,
      self._last_command.coefficients(),
      lat_ctl_limit,
    )
    command = LateralPathCommand(
      valid=valid,
      path_offset=coefficients[0],
      path_angle=coefficients[1],
      curvature=coefficients[2],
      curvature_rate=coefficients[3],
    )
    self._last_command = command
    # The next attack limit must start at the command the PSCM actually saw.
    # This only differs from the allocated targets when limit feedback forced
    # us to retain the previous command.
    self._last_c2 = command.curvature
    self._last_c3 = command.curvature_rate
    return command
