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
PATH_C2_MANEUVER_BP = (0.006, 0.012)
PATH_C2_UNDERTRACKING_BP = (0.006, 0.009)
PATH_C3_MANEUVER_BP = (0.003, 0.006)
PATH_PREVIEW_GEOMETRY_BP = (0.0015, 0.0045)
PATH_TRACKING_ERROR_BP = (0.003, 0.006)
PATH_FIT_HORIZONS = (3.0, 5.0, 7.0, 10.0)


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


def _fit_box_coefficients(rows: list[tuple[float, float, float, float]],
                          lower: tuple[float, float],
                          upper: tuple[float, float]) -> tuple[float, float]:
  """Solve a weighted two-variable least-squares fit inside a box."""
  candidates: list[tuple[float, float]] = []
  s00 = sum(weight * a0 * a0 for weight, a0, _, _ in rows)
  s01 = sum(weight * a0 * a1 for weight, a0, a1, _ in rows)
  s11 = sum(weight * a1 * a1 for weight, _, a1, _ in rows)
  t0 = sum(weight * a0 * target for weight, a0, _, target in rows)
  t1 = sum(weight * a1 * target for weight, _, a1, target in rows)
  determinant = s00 * s11 - s01 ** 2
  if abs(determinant) > 1e-12:
    solution = (
      (t0 * s11 - t1 * s01) / determinant,
      (t1 * s00 - t0 * s01) / determinant,
    )
    if all(lower[index] <= solution[index] <= upper[index] for index in (0, 1)):
      candidates.append(solution)

  for fixed_index in (0, 1):
    free_index = 1 - fixed_index
    for fixed in (lower[fixed_index], upper[fixed_index]):
      denominator = sum(weight * row[free_index] ** 2 for weight, *row, _ in rows)
      numerator = sum(
        weight * row[free_index] * (target - row[fixed_index] * fixed)
        for weight, *row, target in rows
      )
      free = 0.0 if denominator == 0.0 else numerator / denominator
      values = [0.0, 0.0]
      values[fixed_index] = fixed
      values[free_index] = _clip(free, (lower[free_index], upper[free_index]))
      candidates.append((values[0], values[1]))

  candidates.extend(
    (c0, c1)
    for c0 in (lower[0], upper[0])
    for c1 in (lower[1], upper[1])
  )
  return min(
    candidates,
    key=lambda values: sum(
      weight * (a0 * values[0] + a1 * values[1] - target) ** 2
      for weight, a0, a1, target in rows
    ),
  )


def _fit_fast_coefficients(desired: tuple[float, float, float, float],
                           allocated_c2: float, allocated_c3: float,
                           anchor_distance: float,
                           direction: float) -> tuple[float, float]:
  """Fit C0/C1 across space while preserving the active preview target."""
  rows = []
  for distance in PATH_FIT_HORIZONS:
    a0, a1, _, _ = _basis(distance)
    target = _equivalent_curvature(desired, distance) - allocated_c2 - allocated_c3 * distance / 3.0
    rows.append((PATH_MIN_PREVIEW_DISTANCE / distance, a0, a1, target))

  lower = [PATH_LIMITS[0][0], PATH_LIMITS[1][0]]
  upper = [PATH_LIMITS[0][1], PATH_LIMITS[1][1]]
  if direction > 0.0:
    lower = [max(value, 0.0) for value in lower]
  elif direction < 0.0:
    upper = [min(value, 0.0) for value in upper]

  anchor_a0, anchor_a1, _, _ = _basis(anchor_distance)
  anchor_target = _equivalent_curvature(desired, anchor_distance) - \
                  allocated_c2 - allocated_c3 * anchor_distance / 3.0
  c1_lower = max(lower[1], (anchor_target - anchor_a0 * upper[0]) / anchor_a1)
  c1_upper = min(upper[1], (anchor_target - anchor_a0 * lower[0]) / anchor_a1)
  if c1_lower <= c1_upper:
    constant_scale = anchor_target / anchor_a0
    anchor_ratio = anchor_a1 / anchor_a0
    numerator = 0.0
    denominator = 0.0
    for weight, a0, a1, target in rows:
      constant = a0 * constant_scale
      slope = a1 - a0 * anchor_ratio
      numerator += weight * slope * (target - constant)
      denominator += weight * slope ** 2
    c1 = _clip(
      0.0 if denominator == 0.0 else numerator / denominator,
      (c1_lower, c1_upper),
    )
    c0 = (anchor_target - anchor_a1 * c1) / anchor_a0
    return c0, c1

  # An exact anchor can become infeasible when a clipped C3 alone exceeds the
  # target while C0/C1 are constrained to reinforce the requested direction.
  # In that case return the closest bounded spatial fit instead of opposing it.
  return _fit_box_coefficients(rows, (lower[0], lower[1]), (upper[0], upper[1]))


def _bound_nonfast_anchor(desired: tuple[float, float, float, float],
                          allocated_c2: float, allocated_c3: float,
                          anchor_distance: float) -> tuple[float, float]:
  """Keep allocated C2/C3 from overrunning the active-preview target."""
  anchor_target = _equivalent_curvature(desired, anchor_distance)
  if abs(anchor_target) < 1e-12:
    return allocated_c2, allocated_c3

  direction = math.copysign(1.0, anchor_target)
  nonfast_anchor = allocated_c2 + allocated_c3 * anchor_distance / 3.0
  excess = direction * nonfast_anchor - abs(anchor_target)
  if excess <= 0.0:
    return allocated_c2, allocated_c3

  # C3 is the clipped, maneuver-only geometry term, so give it back first.
  c3_anchor = direction * allocated_c3 * anchor_distance / 3.0
  c3_reduction = min(excess, max(c3_anchor, 0.0))
  allocated_c3 -= direction * c3_reduction * 3.0 / anchor_distance
  excess -= c3_reduction

  # C2 can also overrun the anchor around a mixed-sign transition.
  c2_reduction = min(excess, max(direction * allocated_c2, 0.0))
  allocated_c2 -= direction * c2_reduction
  return allocated_c2, allocated_c3


def _conservative_tracking_error(target: float, measured: float, projected: float) -> float:
  """Use projected wheel motion only while it approaches target without crossing."""
  measured_error = target - measured
  projected_error = target - projected
  projected_motion = projected - measured
  if measured_error == 0.0 or projected_motion * measured_error <= 0.0:
    return measured_error
  if projected_error * measured_error <= 0.0:
    return 0.0
  return projected_error if abs(projected_error) < abs(measured_error) else measured_error


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

  def _reset(self) -> LateralPathCommand:
    self._last_command = LateralPathCommand()
    return self._last_command

  def update(self, path, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool,
             projected_measured_curvature: float | None = None,
             desired_angle_curvature: float | None = None,
             lat_ctl_limit: int = 0) -> LateralPathCommand:
    measured_curvature = _finite(measured_curvature)
    projected_curvature = measured_curvature if projected_measured_curvature is None else \
                          _finite(projected_measured_curvature, measured_curvature)
    desired_angle_curvature = None if desired_angle_curvature is None else \
                              _finite(desired_angle_curvature)
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
      distance = _preview_distance(v_ego)
      command = LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured_curvature * distance ** 2, PATH_LIMITS[0]),
        path_angle=_clip(measured_curvature * distance, PATH_LIMITS[1]),
      )
      self._last_command = command
      return command

    if not valid:
      command = LateralPathCommand(
        curvature=_clip(raw[2], PATH_LIMITS[2]),
      )
      self._last_command = command
      return command

    distance = _preview_distance(v_ego)
    target_equivalent = _equivalent_curvature(raw, distance)
    preview_equivalent = target_equivalent - raw[2]
    tracking_error = _conservative_tracking_error(
      target_equivalent,
      measured_curvature,
      projected_curvature,
    )

    maneuver_share = max(
      _interp(abs(raw[2]), *PATH_C2_MANEUVER_BP),
      _interp(abs(raw[3]) * distance / 3.0, *PATH_C3_MANEUVER_BP),
    )
    preview_magnitude_share = _interp(
      abs(preview_equivalent),
      *PATH_PREVIEW_GEOMETRY_BP,
    )
    preview_support = preview_magnitude_share \
      if preview_equivalent * tracking_error > 0.0 else 0.0
    release_share = preview_support * _interp(
      abs(tracking_error),
      *PATH_TRACKING_ERROR_BP,
    )
    undertracking_share = 0.0
    if tracking_error * target_equivalent > 0.0:
      undertracking_share = min(
        _interp(abs(raw[2]), *PATH_C2_UNDERTRACKING_BP),
        _interp(abs(tracking_error), *PATH_TRACKING_ERROR_BP),
      )
    ownership_share = max(maneuver_share, release_share, undertracking_share)

    # C2 owns ordinary driving and is exactly zero at full maneuver ownership.
    # C3 is spatial geometry only; it never carries action/model disagreement.
    allocated_c2 = _clip(raw[2] * (1.0 - ownership_share), PATH_LIMITS[2])
    allocated_c3 = _clip(raw[3] * maneuver_share, PATH_LIMITS[3])

    # Fade model preview geometry independently from coefficient ownership.
    # This keeps tiny ordinary model noise out of C0/C1 without coupling the
    # requested path to measured wheel error.
    geometry_share = max(maneuver_share, preview_magnitude_share)
    desired_path = (
      raw[0] * geometry_share,
      raw[1] * geometry_share,
      raw[2],
      raw[3] * geometry_share,
    )
    allocated_c2, allocated_c3 = _bound_nonfast_anchor(
      desired_path,
      allocated_c2,
      allocated_c3,
      distance,
    )

    # Fit the fast coefficients across space while matching the active preview
    # exactly. Both coefficients reinforce the maneuver instead of canceling
    # one another to reproduce an unclippable polynomial.
    c0, c1 = _fit_fast_coefficients(
      desired_path,
      allocated_c2,
      allocated_c3,
      distance,
      target_equivalent,
    )

    # Measured steering may only extend the complete model path while both the
    # measured and projected wheel remain behind the current desired angle.
    # It can never subtract from or reverse the model command.
    feedback = 0.0
    if desired_angle_curvature is not None and lat_ctl_limit not in (1, 2):
      angle_error = _conservative_tracking_error(
        desired_angle_curvature,
        measured_curvature,
        projected_curvature,
      )
      if angle_error * desired_angle_curvature > 0.0 and \
         angle_error * target_equivalent > 0.0:
        feedback = math.copysign(
          ownership_share * min(abs(angle_error), abs(desired_angle_curvature)),
          angle_error,
        )
    c0 += 0.5 * feedback / _basis(distance)[0]
    c1 += 0.5 * feedback / _basis(distance)[1]
    coefficients = tuple(
      _clip(value, limits)
      for value, limits in zip(
        (c0, c1, allocated_c2, allocated_c3),
        PATH_LIMITS,
        strict=True,
      )
    )

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
    return command
