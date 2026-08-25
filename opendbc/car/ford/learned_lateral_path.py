from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


COEFFICIENT_LIMITS = np.asarray((
  (-5.12, 5.11),
  (-0.5, 0.5235),
  (-0.02, 0.02),
  (-0.001024, 0.001023),
))
COEFFICIENT_RESOLUTIONS = np.asarray((0.01, 0.0005, 0.00002, 0.000001))
COEFFICIENT_SCALES = np.asarray((5.12, 0.5235, 0.02, 0.001024))
PATH_CURVATURE_DISTANCES = (3.0, 5.0, 7.0, 10.0)
FORD_PATH_DT = 0.05
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.35


def driver_steering_opposes_command(steering_pressed: bool, steering_torque: float,
                                     steering_angle_error_deg: float) -> bool:
  if not steering_pressed:
    return False
  if steering_angle_error_deg == 0.0:
    return True
  return steering_torque * steering_angle_error_deg < 0.0


class SteeringAngleProjector:
  def __init__(self, sample_dt: float = FORD_PATH_DT,
               horizon: float = FORD_PATH_ANGLE_PROJECTION_HORIZON):
    self.sample_dt = max(float(sample_dt), FORD_PATH_DT)
    self.horizon = max(float(horizon), 0.0)
    self.samples: deque[float] = deque(maxlen=max(round(self.horizon / self.sample_dt) + 1, 2))
    self.rate_deg_s = 0.0

  def update(self, actual_angle_deg: float) -> float:
    previous_angle_deg = self.samples[-1] if self.samples else float(actual_angle_deg)
    self.samples.append(float(actual_angle_deg))
    self.rate_deg_s = (float(actual_angle_deg) - previous_angle_deg) / self.sample_dt
    sample_time = (len(self.samples) - 1) * self.sample_dt
    if sample_time <= 0.0:
      return float(actual_angle_deg)
    steering_rate_deg_s = (self.samples[-1] - self.samples[0]) / sample_time
    return float(actual_angle_deg) + steering_rate_deg_s * self.horizon


@dataclass(frozen=True)
class PathPolynomial:
  """A Ford/Openpilot lateral path expressed as y, heading, curvature, and spatial curvature slope."""

  c0: float
  c1: float
  c2: float
  c3: float

  def coefficients(self) -> tuple[float, float, float, float]:
    return self.c0, self.c1, self.c2, self.c3

  def as_wire_coefficients(self) -> tuple[float, float, float, float]:
    """Ford's LMC2 CAN signal convention is opposite openpilot's path convention."""
    return tuple(-value for value in self.coefficients())

  def advanced(self, distance_m: float) -> "PathPolynomial":
    """Translate this cubic path into a coordinate frame `distance_m` farther along it."""
    x = max(float(distance_m), 0.0)
    return PathPolynomial(
      self.c0 + self.c1 * x + 0.5 * self.c2 * x**2 + self.c3 * x**3 / 6.0,
      self.c1 + self.c2 * x + 0.5 * self.c3 * x**2,
      self.c2 + self.c3 * x,
      self.c3,
    )

  def horizons(self, speed_mps: float, timestep_s: float, steps: int) -> tuple["PathPolynomial", ...]:
    distance_per_step = max(float(speed_mps), 0.0) * float(timestep_s)
    return tuple(self.advanced(distance_per_step * step) for step in range(1, steps + 1))

  def desired_angles_deg(self, speed_mps: float, timestep_s: float, steps: int,
                         current_desired_angle_deg: float,
                         curvature_to_angle_deg: Callable[[float], float]) -> tuple[float, ...]:
    current_path_angle_deg = curvature_to_angle_deg(-self.c2)
    return tuple(
      current_desired_angle_deg + curvature_to_angle_deg(-path.c2) - current_path_angle_deg
      for path in self.horizons(speed_mps, timestep_s, steps)
    )


@dataclass(frozen=True)
class LearnedLateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0

  def coefficients(self) -> tuple[float, float, float, float]:
    return self.path_offset, self.path_angle, self.curvature, self.curvature_rate


def _equivalent_curvature(coefficients: np.ndarray, distance_m: float) -> float:
  c0, c1, c2, c3 = coefficients
  offset = c0 + c1 * distance_m + 0.5 * c2 * distance_m**2 + c3 * distance_m**3 / 6.0
  return float(2.0 * offset / distance_m**2)


def lmc2_control_utilization(command: LearnedLateralPathCommand, lat_ctl_limit: int) -> float:
  """Return signed command-envelope usage, raised by genuine PSCM limit status."""
  coefficients = np.asarray(command.coefficients())
  denominators = np.where(coefficients >= 0.0, COEFFICIENT_LIMITS[:, 1], np.abs(COEFFICIENT_LIMITS[:, 0]))
  coefficient_utilization = np.abs(coefficients) / denominators
  coefficient_magnitude = float(np.clip(np.max(coefficient_utilization), 0.0, 1.0))
  if coefficient_magnitude == 0.0:
    return 0.0

  magnitude = coefficient_magnitude
  if lat_ctl_limit == 1:
    magnitude = max(magnitude, 0.8)
  elif lat_ctl_limit == 2:
    magnitude = 1.0

  direction_source = _equivalent_curvature(coefficients, 7.0)
  if abs(direction_source) < 1e-9:
    direction_source = float(coefficients[int(np.argmax(coefficient_utilization))])
  return math.copysign(magnitude, direction_source)


class LearnedLateralPathController:
  """Compact, memoryless Ford LMC2 policy distilled from validated road control."""

  def __init__(self, model_path: Path | None = None):
    path = Path(__file__).with_name("ford_lateral_policy_v1.npz") if model_path is None else model_path
    with np.load(path, allow_pickle=False) as model:
      self.l1_weight = model["l1.weight"]
      self.l1_bias = model["l1.bias"]
      self.l2_weight = model["l2.weight"]
      self.l2_bias = model["l2.bias"]
      self.out_weight = model["out.weight"]
      self.out_bias = model["out.bias"]

  @staticmethod
  def _features(path: PathPolynomial, desired_angle_deg: float, actual_angle_deg: float,
                steering_rate_deg_s: float, speed_mps: float, eps_current_a: float,
                projected_curvature: float, measured_curvature: float,
                desired_curvature: float, lat_ctl_limit: int, active: bool) -> np.ndarray:
    coefficients = np.asarray(path.coefficients())
    normalized_path = np.clip(coefficients / COEFFICIENT_SCALES, -2.0, 2.0)
    path_curvatures = np.clip(np.asarray([
      _equivalent_curvature(coefficients, distance) for distance in PATH_CURVATURE_DISTANCES
    ]) / 0.08, -3.0, 3.0)
    return np.concatenate((
      normalized_path,
      path_curvatures,
      np.asarray((
        np.clip(desired_angle_deg / 300.0, -3.0, 3.0),
        np.clip(actual_angle_deg / 300.0, -3.0, 3.0),
        np.clip((desired_angle_deg - actual_angle_deg) / 150.0, -3.0, 3.0),
        np.clip(steering_rate_deg_s / 300.0, -3.0, 3.0),
        np.clip(speed_mps / 30.0, 0.0, 3.0),
        np.clip(eps_current_a / 50.0, -3.0, 3.0),
        np.clip(projected_curvature / 0.05, -3.0, 3.0),
        np.clip(measured_curvature / 0.05, -3.0, 3.0),
        np.clip(desired_curvature / 0.05, -3.0, 3.0),
        np.clip((desired_curvature - projected_curvature) / 0.05, -3.0, 3.0),
        np.clip(float(lat_ctl_limit) / 2.0, 0.0, 1.0),
        float(active),
      )),
    ))

  @staticmethod
  def _quantize_wire(coefficients: np.ndarray) -> np.ndarray:
    clipped = np.clip(coefficients, COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1])
    steps = np.rint((clipped - COEFFICIENT_LIMITS[:, 0]) / COEFFICIENT_RESOLUTIONS)
    return np.clip(
      COEFFICIENT_LIMITS[:, 0] + steps * COEFFICIENT_RESOLUTIONS,
      COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1],
    )

  def update(self, path, desired_angle_deg: float, actual_angle_deg: float,
             steering_rate_deg_s: float, speed_mps: float, eps_current_a: float,
             projected_curvature: float, measured_curvature: float,
             desired_curvature: float, lat_ctl_limit: int, active: bool) -> LearnedLateralPathCommand:
    if not active or path is None or not bool(getattr(path, "valid", False)):
      return LearnedLateralPathCommand()

    polynomial = PathPolynomial(
      float(path.pathOffset), float(path.pathAngle), float(path.curvature), float(path.curvatureRate),
    )
    features = self._features(
      polynomial, desired_angle_deg, actual_angle_deg, steering_rate_deg_s, speed_mps, eps_current_a,
      projected_curvature, measured_curvature, desired_curvature, lat_ctl_limit, active,
    )
    hidden = np.tanh(self.l1_weight @ features + self.l1_bias)
    hidden = np.tanh(self.l2_weight @ hidden + self.l2_bias)
    wire_coefficients = self._quantize_wire(
      np.tanh(self.out_weight @ hidden + self.out_bias) * COEFFICIENT_SCALES,
    )
    # CarController retains the normal openpilot convention and negates once
    # at the CAN boundary, so convert the learned wire command back here.
    command = -wire_coefficients
    return LearnedLateralPathCommand(True, *(float(value) for value in command))
