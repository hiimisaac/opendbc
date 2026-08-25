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
COMMAND_LIMITS = np.column_stack((-COEFFICIENT_LIMITS[:, 1], -COEFFICIENT_LIMITS[:, 0]))
COEFFICIENT_RESOLUTIONS = np.asarray((0.01, 0.0005, 0.00002, 0.000001))
COEFFICIENT_SCALES = np.asarray((5.12, 0.5235, 0.02, 0.001024))
PATH_CURVATURE_DISTANCES = (3.0, 5.0, 7.0, 10.0)
FORD_PATH_DT = 0.05
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.35
ADAPTIVE_RATE = 0.08
ADAPTIVE_LEAK = 0.01
ADAPTIVE_GAIN_LIMIT = 0.12
ADAPTIVE_MANEUVER_START = 0.002
ADAPTIVE_MANEUVER_FULL = 0.012
MODEL_C2_FULL_OWNERSHIP_ANGLE_DEG = 35.0
MODEL_C2_ZERO_OWNERSHIP_ANGLE_DEG = 80.0
MODEL_C2_FULL_OWNERSHIP_CURVATURE = 0.008
MODEL_C2_ZERO_OWNERSHIP_CURVATURE = 0.018
OUTCOME_CORRECTION_START_ANGLE_DEG = 10.0
OUTCOME_CORRECTION_FULL_ANGLE_DEG = 25.0
OUTCOME_CORRECTION_START_ERROR_DEG = 4.0
OUTCOME_CORRECTION_FULL_ERROR_DEG = 14.0
OUTCOME_CORRECTION_FULL_SPEED_MPS = 16.0
OUTCOME_CORRECTION_ZERO_SPEED_MPS = 18.0
FAST_COEFFICIENT_INDICES = np.asarray((0, 1, 3), dtype=np.int64)
FORD_LATERAL_POLICY_VERSION = 2
FORD_LATERAL_POLICY_ARRAYS = frozenset((
  "l1.weight", "l1.bias", "l2.weight", "l2.bias", "out.weight", "out.bias",
  "residual.l1.weight", "residual.l1.bias", "residual.out.weight", "residual.out.bias", "residual.scales",
))


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
    c0, c1, c2, c3 = self.coefficients()
    return -c0, -c1, -c2, -c3

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


@dataclass(frozen=True)
class AdaptiveLateralState:
  enabled: bool = False
  gain: float = 0.0
  reference_curvature: float = 0.0
  tracking_error: float = 0.0
  adapting: bool = False


class AdaptiveLateralTrim:
  """Bounded per-drive adaptation around a nominal Ford LMC2 command."""

  def __init__(self, enabled: bool = False):
    self.enabled = bool(enabled)
    self._gains = np.zeros(2)
    self.state = AdaptiveLateralState(enabled=self.enabled)

  @staticmethod
  def _maneuver_weight(desired_curvature: float) -> float:
    normalized = float(np.clip(
      (abs(desired_curvature) - ADAPTIVE_MANEUVER_START) /
      (ADAPTIVE_MANEUVER_FULL - ADAPTIVE_MANEUVER_START), 0.0, 1.0,
    ))
    return normalized**2 * (3.0 - 2.0 * normalized)

  def set_enabled(self, enabled: bool) -> None:
    enabled = bool(enabled)
    if self.enabled and not enabled:
      self._gains.fill(0.0)
    self.enabled = enabled
    self.state = AdaptiveLateralState(enabled=enabled)

  def update(self, nominal: LearnedLateralPathCommand, desired_curvature: float,
             measured_curvature: float, active: bool, driver_input: bool,
             lat_ctl_limit: int, projected_curvature: float | None = None) -> LearnedLateralPathCommand:
    if not self.enabled:
      return nominal
    if not active or not nominal.valid:
      self.state = AdaptiveLateralState(enabled=True)
      return nominal

    reference_curvature = float(desired_curvature)
    arrival_curvature = float(measured_curvature if projected_curvature is None else projected_curvature)
    tracking_error = reference_curvature - arrival_curvature

    coefficients = np.asarray(nominal.coefficients())
    fast_coefficients = coefficients.copy()
    fast_coefficients[2] = 0.0
    fast_curvature = _equivalent_curvature(fast_coefficients, 7.0)
    maneuver_weight = self._maneuver_weight(desired_curvature)
    direction_agrees = desired_curvature * fast_curvature > 0.0
    direction_index = int(desired_curvature >= 0.0)
    adaptation_frozen = driver_input or lat_ctl_limit != 0
    adapting = bool(direction_agrees and maneuver_weight > 0.0 and not adaptation_frozen)

    if not adaptation_frozen:
      self._gains *= max(1.0 - ADAPTIVE_LEAK * FORD_PATH_DT, 0.0)
    if adapting:
      normalized_error = float(np.clip(tracking_error / 0.02, -1.0, 1.0))
      normalized_command = float(np.clip(fast_curvature / 0.1, -1.0, 1.0))
      self._gains[direction_index] = np.clip(
        self._gains[direction_index] +
        ADAPTIVE_RATE * FORD_PATH_DT * normalized_error * normalized_command * maneuver_weight,
        -ADAPTIVE_GAIN_LIMIT, ADAPTIVE_GAIN_LIMIT,
      )

    applied_gain = float(self._gains[direction_index]) if direction_agrees else 0.0
    requested_scale = 1.0 + applied_gain * maneuver_weight
    maximum_scale = 1.0 + ADAPTIVE_GAIN_LIMIT
    for index in (0, 1, 3):
      value = coefficients[index]
      if value > 0.0:
        maximum_scale = min(maximum_scale, COMMAND_LIMITS[index, 1] / value)
      elif value < 0.0:
        maximum_scale = min(maximum_scale, COMMAND_LIMITS[index, 0] / value)
    scale = float(np.clip(requested_scale, 1.0 - ADAPTIVE_GAIN_LIMIT, maximum_scale))
    adapted = coefficients.copy()
    adapted[[0, 1, 3]] *= scale
    self.state = AdaptiveLateralState(
      enabled=True, gain=(scale - 1.0), reference_curvature=reference_curvature,
      tracking_error=tracking_error, adapting=adapting,
    )
    return LearnedLateralPathCommand(nominal.valid, *(float(value) for value in adapted))


def _equivalent_curvature(coefficients: np.ndarray, distance_m: float) -> float:
  c0, c1, c2, c3 = coefficients
  offset = c0 + c1 * distance_m + 0.5 * c2 * distance_m**2 + c3 * distance_m**3 / 6.0
  return float(2.0 * offset / distance_m**2)


def _quantize_wire_coefficients(coefficients: np.ndarray) -> np.ndarray:
  clipped = np.clip(coefficients, COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1])
  steps = np.rint((clipped - COEFFICIENT_LIMITS[:, 0]) / COEFFICIENT_RESOLUTIONS)
  return np.clip(
    COEFFICIENT_LIMITS[:, 0] + steps * COEFFICIENT_RESOLUTIONS,
    COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1],
  )


def _model_c2_ownership(path: PathPolynomial, desired_angle_deg: float) -> float:
  """Keep Ford's stable C2 on ordinary roads, then stop feeding its sticky PSCM path in large turns."""
  angle_progress = (
    (abs(desired_angle_deg) - MODEL_C2_FULL_OWNERSHIP_ANGLE_DEG) /
    (MODEL_C2_ZERO_OWNERSHIP_ANGLE_DEG - MODEL_C2_FULL_OWNERSHIP_ANGLE_DEG)
  )
  curvature_progress = (
    (abs(path.c2) - MODEL_C2_FULL_OWNERSHIP_CURVATURE) /
    (MODEL_C2_ZERO_OWNERSHIP_CURVATURE - MODEL_C2_FULL_OWNERSHIP_CURVATURE)
  )
  progress = float(np.clip(max(angle_progress, curvature_progress), 0.0, 1.0))
  return 1.0 - progress**2 * (3.0 - 2.0 * progress)


def _smoothstep(value: float, start: float, end: float) -> float:
  progress = float(np.clip((value - start) / (end - start), 0.0, 1.0))
  return progress**2 * (3.0 - 2.0 * progress)


def _outcome_correction_weight(path: PathPolynomial, desired_angle_deg: float,
                               actual_angle_deg: float, steering_rate_deg_s: float,
                               speed_mps: float, projected_curvature: float, desired_curvature: float,
                               driver_input: bool, lat_ctl_limit: int) -> float:
  """Continuously admit learned authority only while the wheel is demonstrably behind the model."""
  if driver_input or lat_ctl_limit != 0 or desired_angle_deg == 0.0:
    return 0.0
  model_curvature = _equivalent_curvature(np.asarray(path.coefficients()), 7.0)
  if desired_angle_deg * model_curvature >= 0.0:
    return 0.0
  projected_angle_deg = actual_angle_deg + steering_rate_deg_s * FORD_PATH_ANGLE_PROJECTION_HORIZON
  direction = math.copysign(1.0, desired_angle_deg)
  actual_undertrack_deg = direction * (desired_angle_deg - actual_angle_deg)
  projected_undertrack_deg = direction * (desired_angle_deg - projected_angle_deg)
  if desired_curvature != 0.0:
    curvature_direction = math.copysign(1.0, desired_curvature)
    if curvature_direction * (desired_curvature - projected_curvature) <= 0.0:
      return 0.0
  # Both estimates must still be behind. This rejects a fast unwind that has
  # already crossed the target, even if the projection lands on its far side.
  conservative_undertrack_deg = min(actual_undertrack_deg, projected_undertrack_deg)
  speed_weight = 1.0 - _smoothstep(
    speed_mps, OUTCOME_CORRECTION_FULL_SPEED_MPS, OUTCOME_CORRECTION_ZERO_SPEED_MPS,
  )
  return (
    _smoothstep(abs(desired_angle_deg), OUTCOME_CORRECTION_START_ANGLE_DEG, OUTCOME_CORRECTION_FULL_ANGLE_DEG) *
    _smoothstep(conservative_undertrack_deg, OUTCOME_CORRECTION_START_ERROR_DEG, OUTCOME_CORRECTION_FULL_ERROR_DEG) *
    speed_weight
  )


def _apply_model_aligned_outcome_residual(wire_command: np.ndarray, model_wire: np.ndarray,
                                          residual: np.ndarray, correction_weight: float) -> np.ndarray:
  """Add 7 m authority while keeping the complete bounded arc on the model-requested side."""
  bounded_baseline = np.clip(wire_command, COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1])
  full_candidate = wire_command.copy()
  full_candidate[FAST_COEFFICIENT_INDICES] += correction_weight * residual
  bounded_candidate = np.clip(full_candidate, COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1])
  bounded_delta = bounded_candidate - bounded_baseline

  model_7m = _equivalent_curvature(model_wire, 7.0)
  added_7m = _equivalent_curvature(bounded_delta, 7.0)
  if model_7m * added_7m <= 0.0:
    return wire_command

  scale = 1.0
  for distance in PATH_CURVATURE_DISTANCES:
    model_curvature = _equivalent_curvature(model_wire, distance)
    if abs(model_curvature) < 1e-9:
      continue
    direction = math.copysign(1.0, model_curvature)
    baseline_alignment = direction * _equivalent_curvature(bounded_baseline, distance)
    delta_alignment = direction * _equivalent_curvature(bounded_delta, distance)
    if baseline_alignment <= 0.0:
      if delta_alignment <= 0.0:
        return wire_command
      continue
    if delta_alignment < 0.0:
      scale = min(scale, baseline_alignment / -delta_alignment)

  quantized_baseline = _quantize_wire_coefficients(bounded_baseline)

  def quantized_candidate(candidate_scale: float) -> np.ndarray:
    return _quantize_wire_coefficients(bounded_baseline + candidate_scale * bounded_delta)

  def arc_is_safe(candidate: np.ndarray) -> bool:
    for distance in PATH_CURVATURE_DISTANCES:
      model_curvature = _equivalent_curvature(model_wire, distance)
      if abs(model_curvature) < 1e-9:
        continue
      baseline_alignment = model_curvature * _equivalent_curvature(quantized_baseline, distance)
      candidate_alignment = model_curvature * _equivalent_curvature(candidate, distance)
      if baseline_alignment >= 0.0 and candidate_alignment < 0.0:
        return False
    return model_7m * (
      _equivalent_curvature(candidate, 7.0) - _equivalent_curvature(quantized_baseline, 7.0)
    ) > 0.0

  scale = float(np.clip(scale, 0.0, 1.0))
  for candidate_scale in np.linspace(scale, 0.0, 33)[:-1]:
    if arc_is_safe(quantized_candidate(float(candidate_scale))):
      return bounded_baseline + float(candidate_scale) * bounded_delta
  return wire_command


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
    path = Path(__file__).with_name("ford_lateral_policy_v2.npz") if model_path is None else model_path
    with np.load(path, allow_pickle=False) as model:
      if "version" not in model.files:
        raise ValueError("Ford lateral policy artifact is missing its schema version")
      version = int(model["version"])
      if version != FORD_LATERAL_POLICY_VERSION:
        raise ValueError(f"Unsupported Ford lateral policy version {version}")
      missing_arrays = FORD_LATERAL_POLICY_ARRAYS.difference(model.files)
      if missing_arrays:
        raise ValueError(f"Ford lateral policy v{version} is missing arrays: {sorted(missing_arrays)}")
      self.l1_weight = model["l1.weight"]
      self.l1_bias = model["l1.bias"]
      self.l2_weight = model["l2.weight"]
      self.l2_bias = model["l2.bias"]
      self.out_weight = model["out.weight"]
      self.out_bias = model["out.bias"]
      self.residual_l1_weight = model["residual.l1.weight"]
      self.residual_l1_bias = model["residual.l1.bias"]
      self.residual_out_weight = model["residual.out.weight"]
      self.residual_out_bias = model["residual.out.bias"]
      self.residual_scales = model["residual.scales"]
    self.adaptive_trim = AdaptiveLateralTrim()

  @property
  def adaptive_state(self) -> AdaptiveLateralState:
    return self.adaptive_trim.state

  def set_adaptive_enabled(self, enabled: bool) -> None:
    self.adaptive_trim.set_enabled(enabled)

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
    return _quantize_wire_coefficients(coefficients)

  def update(self, path, desired_angle_deg: float, actual_angle_deg: float,
             steering_rate_deg_s: float, speed_mps: float, eps_current_a: float,
             projected_curvature: float, measured_curvature: float,
             desired_curvature: float, lat_ctl_limit: int, active: bool,
             driver_input: bool = False) -> LearnedLateralPathCommand:
    if not active or path is None or not bool(getattr(path, "valid", False)):
      inactive_command = LearnedLateralPathCommand()
      self.adaptive_trim.update(
        inactive_command, desired_curvature, measured_curvature, False, driver_input, lat_ctl_limit,
        projected_curvature=projected_curvature,
      )
      return inactive_command

    polynomial = PathPolynomial(
      float(path.pathOffset), float(path.pathAngle), float(path.curvature), float(path.curvatureRate),
    )
    features = self._features(
      polynomial, desired_angle_deg, actual_angle_deg, steering_rate_deg_s, speed_mps, eps_current_a,
      projected_curvature, measured_curvature, desired_curvature, lat_ctl_limit, active,
    )
    hidden = np.tanh(self.l1_weight @ features + self.l1_bias)
    hidden = np.tanh(self.l2_weight @ hidden + self.l2_bias)
    wire_command = np.tanh(self.out_weight @ hidden + self.out_bias) * COEFFICIENT_SCALES
    wire_command[2] = -polynomial.c2 * _model_c2_ownership(polynomial, desired_angle_deg)
    correction_weight = _outcome_correction_weight(
      polynomial, desired_angle_deg, actual_angle_deg, steering_rate_deg_s, speed_mps,
      projected_curvature, desired_curvature, driver_input, lat_ctl_limit,
    )
    if correction_weight > 0.0:
      residual_hidden = np.tanh(self.residual_l1_weight @ features + self.residual_l1_bias)
      residual = np.tanh(self.residual_out_weight @ residual_hidden + self.residual_out_bias) * self.residual_scales
      model_wire = -np.asarray(polynomial.coefficients())
      wire_command = _apply_model_aligned_outcome_residual(wire_command, model_wire, residual, correction_weight)
    wire_coefficients = self._quantize_wire(wire_command)
    # CarController retains the normal openpilot convention and negates once
    # at the CAN boundary, so convert the learned wire command back here.
    command = -wire_coefficients
    nominal = LearnedLateralPathCommand(True, *(float(value) for value in command))
    adapted = self.adaptive_trim.update(
      nominal, desired_curvature, measured_curvature, active, driver_input, lat_ctl_limit,
      projected_curvature=projected_curvature,
    )
    if adapted == nominal:
      return nominal
    adapted_wire = self._quantize_wire(-np.asarray(adapted.coefficients()))
    return LearnedLateralPathCommand(True, *(float(value) for value in -adapted_wire))
