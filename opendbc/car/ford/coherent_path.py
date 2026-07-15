from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np


# LateralMotionControl2 describes one path polynomial:
#   y(s) = c0 + c1*s + 0.5*c2*s**2 + c3*s**3/6
#
# C2 is unusual on the tested Ford PSCM: its wire command has an approximately
# one-second response. C0/C1/C3 act much sooner. Treating the latest C2 wire
# value as though it were already at the wheels makes the four coefficients
# describe different paths during every entry, release, and reversal. This
# controller instead estimates effective C2 and projects the complete desired
# model path onto the three remaining, bounded wire coefficients.

FORD_COHERENT_C0_LIMITS = (-4.61, 4.60)
FORD_COHERENT_C1_LIMITS = (-0.475, 0.497)
FORD_COHERENT_C2_LIMITS = (-0.02, 0.02)
FORD_COHERENT_C3_LIMITS = (-0.001024, 0.001023)

FORD_COHERENT_C2_FADE_BP = (0.006, 0.012)
FORD_COHERENT_C2_ATTACK_STEP = 0.0002
FORD_COHERENT_C2_RESPONSE_TAU = 1.0
FORD_COHERENT_DT = 0.05
FORD_COHERENT_HORIZONS = (0.0, 1.75, 3.5, 5.0, 7.0)
FORD_COHERENT_HEADING_WEIGHT = 3.5
FORD_COHERENT_ANGLE_ERROR_LIMIT = 0.006
FORD_COHERENT_ANGLE_ERROR_DEADZONE_DEG = 2.5

# Handoff uses one blend fraction for the complete polynomial. These limits
# only bound the first model step after a driver releases the wheel; ordinary
# model releases and reversals remain immediate.
FORD_COHERENT_HANDOFF_C0_STEP = 0.245
FORD_COHERENT_HANDOFF_C1_STEP = 0.07
FORD_COHERENT_HANDOFF_C3_STEP = 0.001


@dataclass(frozen=True)
class CoherentPathCommand:
  curvature: float
  curvature_rate: float
  path_angle: float
  path_offset: float
  estimated_c2: float
  cooperative_control: bool


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _valid_model(model) -> bool:
  if model is None:
    return False
  try:
    count = len(model.position.x)
    return count > 1 and count == len(model.position.y) == len(model.orientation.z)
  except (AttributeError, TypeError):
    return False


def _model_samples(model) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  if not _valid_model(model):
    return None

  try:
    xs = np.asarray(model.position.x, dtype=float)
    ys = np.asarray(model.position.y, dtype=float)
    headings = np.unwrap(np.asarray(model.orientation.z, dtype=float))
  except (AttributeError, TypeError, ValueError):
    return None

  if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)) or not np.all(np.isfinite(headings)):
    return None

  spatial_distances = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))))
  unique = np.concatenate(([True], np.diff(spatial_distances) > 1e-9))
  spatial_distances = spatial_distances[unique]
  ys = ys[unique]
  headings = headings[unique]
  if len(spatial_distances) < 2 or spatial_distances[-1] <= 0.0:
    return None

  horizon = min(FORD_COHERENT_HORIZONS[-1], float(spatial_distances[-1]))
  distances = np.asarray(sorted({min(distance, horizon) for distance in FORD_COHERENT_HORIZONS}), dtype=float)
  if len(distances) < 2 or distances[-1] <= 0.0:
    return None

  return distances, np.interp(distances, spatial_distances, ys), np.interp(distances, spatial_distances, headings)


def _path_matrix(distances: np.ndarray) -> np.ndarray:
  position_rows = np.column_stack((np.ones_like(distances), distances, distances ** 3 / 6.0))
  heading_rows = np.column_stack((np.zeros_like(distances), np.ones_like(distances), distances ** 2 / 2.0))
  return np.vstack((position_rows, FORD_COHERENT_HEADING_WEIGHT * heading_rows))


def _fit_free_coefficients(distances: np.ndarray, positions: np.ndarray, headings: np.ndarray,
                           fixed_c2: float) -> np.ndarray:
  matrix = _path_matrix(distances)
  residual_position = positions - 0.5 * fixed_c2 * distances ** 2
  residual_heading = headings - fixed_c2 * distances
  target = np.concatenate((residual_position, FORD_COHERENT_HEADING_WEIGHT * residual_heading))
  coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
  return coefficients


def _fit_bounded_coefficients(distances: np.ndarray, positions: np.ndarray, headings: np.ndarray,
                              fixed_c2: float) -> np.ndarray:
  """Solve the three-variable box-constrained path projection exactly.

  With only C0/C1/C3, enumerating each variable as free/lower/upper is small
  (27 cases), deterministic, and avoids adding a scipy runtime dependency.
  """
  matrix = _path_matrix(distances)
  residual_position = positions - 0.5 * fixed_c2 * distances ** 2
  residual_heading = headings - fixed_c2 * distances
  target = np.concatenate((residual_position, FORD_COHERENT_HEADING_WEIGHT * residual_heading))
  limits = (FORD_COHERENT_C0_LIMITS, FORD_COHERENT_C1_LIMITS, FORD_COHERENT_C3_LIMITS)

  best_coefficients = np.zeros(3)
  best_error = math.inf
  for states in product((-1, 0, 1), repeat=3):
    coefficients = np.zeros(3)
    fixed = [i for i, state in enumerate(states) if state != 0]
    free = [i for i, state in enumerate(states) if state == 0]
    for i in fixed:
      coefficients[i] = limits[i][0 if states[i] < 0 else 1]

    free_target = target - matrix[:, fixed] @ coefficients[fixed] if fixed else target
    if free:
      coefficients[free], *_ = np.linalg.lstsq(matrix[:, free], free_target, rcond=None)

    if any(coefficients[i] < limits[i][0] - 1e-12 or coefficients[i] > limits[i][1] + 1e-12 for i in free):
      continue

    error = float(np.sum((matrix @ coefficients - target) ** 2))
    if error < best_error:
      best_error = error
      best_coefficients = coefficients

  return best_coefficients


def _polynomial_samples(coefficients: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  c0, c1, c2, c3 = coefficients
  positions = c0 + c1 * distances + 0.5 * c2 * distances ** 2 + c3 * distances ** 3 / 6.0
  headings = c1 + c2 * distances + 0.5 * c3 * distances ** 2
  return positions, headings


def _same_direction_attack_limit(value: float, previous: float, step: float) -> float:
  if value * previous >= 0.0 and abs(value) > abs(previous):
    return math.copysign(min(abs(value), abs(previous) + step), value)
  return value


class CoherentPathController:
  def __init__(self) -> None:
    self.estimated_c2 = 0.0
    self.c2_command = 0.0
    self.last_wire_coefficients = np.zeros(3)
    self.handoff_active = False

  @staticmethod
  def _c2_target(desired_curvature: float, driver_override: bool) -> float:
    if driver_override:
      return 0.0
    magnitude = abs(desired_curvature)
    if magnitude <= FORD_COHERENT_C2_FADE_BP[0]:
      share = 1.0
    elif magnitude >= FORD_COHERENT_C2_FADE_BP[1]:
      share = 0.0
    else:
      share = 1.0 - ((magnitude - FORD_COHERENT_C2_FADE_BP[0]) /
                     (FORD_COHERENT_C2_FADE_BP[1] - FORD_COHERENT_C2_FADE_BP[0]))
    return _clip(desired_curvature, FORD_COHERENT_C2_LIMITS) * share

  def _update_c2(self, desired_curvature: float, active: bool, driver_override: bool) -> None:
    # The interval that just elapsed used the previous wire command. Advance
    # plant state before selecting the command that will govern the next one.
    alpha = 1.0 - math.exp(-FORD_COHERENT_DT / FORD_COHERENT_C2_RESPONSE_TAU)
    self.estimated_c2 += alpha * (self.c2_command - self.estimated_c2)
    self.estimated_c2 = _clip(_finite(self.estimated_c2), FORD_COHERENT_C2_LIMITS)

    target = self._c2_target(desired_curvature, driver_override) if active else 0.0
    if target * self.c2_command < 0.0:
      # Flush the old wire sign for one frame. The residual projection already
      # requests the new path immediately through C0/C1/C3.
      self.c2_command = 0.0
    else:
      self.c2_command = _same_direction_attack_limit(target, self.c2_command,
                                                     FORD_COHERENT_C2_ATTACK_STEP)

  @staticmethod
  def _desired_polynomial(model, desired_curvature: float, angle_error_curvature: float,
                          driver_override: bool, measured_curvature: float) -> tuple[np.ndarray, np.ndarray]:
    if driver_override:
      distances = np.asarray(FORD_COHERENT_HORIZONS, dtype=float)
      return distances, np.asarray((0.0, 0.0, measured_curvature, 0.0))

    desired_c2 = desired_curvature + angle_error_curvature
    samples = _model_samples(model)
    if samples is None:
      distances = np.asarray(FORD_COHERENT_HORIZONS, dtype=float)
      return distances, np.asarray((0.0, 0.0, desired_c2, 0.0))

    distances, positions, headings = samples
    # Steering-angle error is a bounded correction to the requested path, not
    # an independent coefficient. Adding its arc to both model observables
    # keeps the constrained action and spatial path internally consistent.
    positions = positions + 0.5 * angle_error_curvature * distances ** 2
    headings = headings + angle_error_curvature * distances
    c0, c1, c3 = _fit_free_coefficients(distances, positions, headings, desired_c2)
    return distances, np.asarray((c0, c1, desired_c2, c3))

  def _handoff_blend(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
    delta = target - self.last_wire_coefficients
    fraction = 1.0
    for difference, step in zip(delta,
                                (FORD_COHERENT_HANDOFF_C0_STEP,
                                 FORD_COHERENT_HANDOFF_C1_STEP,
                                 FORD_COHERENT_HANDOFF_C3_STEP), strict=True):
      if abs(difference) > step:
        fraction = min(fraction, step / abs(difference))
    return self.last_wire_coefficients + fraction * delta, fraction == 1.0

  def update(self, model, desired_curvature: float, measured_curvature: float,
             v_ego: float, lat_active: bool, driver_override: bool,
             angle_error_curvature: float) -> CoherentPathCommand:
    del v_ego  # The polynomial is spatial; no wheelbase- or speed-dependent gain belongs here.
    desired_curvature = _finite(desired_curvature)
    measured_curvature = _finite(measured_curvature)
    angle_error_curvature = _clip(_finite(angle_error_curvature),
                                  (-FORD_COHERENT_ANGLE_ERROR_LIMIT, FORD_COHERENT_ANGLE_ERROR_LIMIT))
    driver_override = bool(lat_active and driver_override)

    self._update_c2(desired_curvature, lat_active, driver_override)
    if not lat_active:
      self.last_wire_coefficients[:] = 0.0
      self.handoff_active = False
      return CoherentPathCommand(0.0, 0.0, 0.0, 0.0, self.estimated_c2, False)

    if driver_override:
      self.handoff_active = True
      angle_error_curvature = 0.0

    distances, desired_coefficients = self._desired_polynomial(
      model, desired_curvature, angle_error_curvature, driver_override, measured_curvature,
    )
    desired_positions, desired_headings = _polynomial_samples(desired_coefficients, distances)
    wire_coefficients = _fit_bounded_coefficients(
      distances, desired_positions, desired_headings, self.estimated_c2,
    )

    cooperative_control = driver_override or self.handoff_active
    if self.handoff_active and not driver_override:
      wire_coefficients, handoff_complete = self._handoff_blend(wire_coefficients)
      if handoff_complete:
        self.handoff_active = False

    self.last_wire_coefficients = wire_coefficients
    c0, c1, c3 = wire_coefficients
    return CoherentPathCommand(self.c2_command, float(c3), float(c1), float(c0),
                               self.estimated_c2, cooperative_control)
