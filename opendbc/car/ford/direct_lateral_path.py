"""Direct Ford LMC2 path adapter for clean-polynomial experiments."""

from dataclasses import dataclass
import math


PATH_MIN_LOOKAHEAD = 7.0
PATH_LIMITS = (
  (-5.11, 5.12),
  (-0.5235, 0.5),
  (-0.02, 0.02),
  (-0.001023, 0.001024),
)


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0

  def coefficients(self) -> tuple[float, float, float, float]:
    return self.path_offset, self.path_angle, self.curvature, self.curvature_rate

def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _equivalent_curvature(coefficients: tuple[float, float, float, float], distance: float) -> float:
  basis = 2.0 / distance ** 2, 2.0 / distance, 1.0, distance / 3.0
  return sum(basis[index] * coefficients[index] for index in range(4))


def lmc2_control_utilization(command: LateralPathCommand, lat_ctl_limit: int) -> float:
  """Return signed LMC2 envelope use, augmented by the PSCM limit status."""
  coefficients = command.coefficients()
  utilization = [
    abs(value) / (limits[1] if value >= 0.0 else abs(limits[0]))
    for value, limits in zip(coefficients, PATH_LIMITS, strict=True)
  ]
  coefficient_magnitude = _clip(max(utilization), (0.0, 1.0))
  if coefficient_magnitude == 0.0:
    return 0.0
  magnitude = max(coefficient_magnitude, 0.8) if lat_ctl_limit == 1 else coefficient_magnitude
  magnitude = 1.0 if lat_ctl_limit == 2 else magnitude
  direction = _equivalent_curvature(coefficients, PATH_MIN_LOOKAHEAD)
  if abs(direction) < 1e-9:
    direction = coefficients[max(range(4), key=utilization.__getitem__)]
  return math.copysign(magnitude, direction)


class DirectLatControlPath:
  """Pass one model-derived cubic to LMC2 without steering-feedback shaping."""

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    del measured_curvature, v_ego, driver_override, projected_measured_curvature, desired_angle_curvature, lat_ctl_limit
    if not active:
      return LateralPathCommand()

    valid = path is not None and bool(getattr(path, "valid", False))
    raw = (
      _finite(getattr(path, "pathOffset", 0.0)) if valid else 0.0,
      _finite(getattr(path, "pathAngle", 0.0)) if valid else 0.0,
      _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0,
      _finite(getattr(path, "curvatureRate", 0.0)) if valid else 0.0,
    )
    coefficients = tuple(
      _clip(value, limits)
      for value, limits in zip(raw, PATH_LIMITS, strict=True)
    )
    return LateralPathCommand(valid, *coefficients)
