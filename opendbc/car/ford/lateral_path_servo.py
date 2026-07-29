"""Ford LMC2 polynomial servo.

C2 owns ordinary driving. Model geometry adds C0/C1/C3 only for spatial
maneuvers; measured steering can extend that command only while both current
and projected wheel positions remain behind the desired angle. A single
terminal invariant removes outward authority after arrival without estimating
PSCM dynamics or rate-limiting the command.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


PATH_LIMITS = (
  (-4.61, 4.60),
  (-0.475, 0.497),
  (-0.02, 0.02),
  (-0.001024, 0.001023),
)
PATH_MIN_LOOKAHEAD = 7.0
PATH_C2_BASEBAND_BP = (0.003, 0.006)
PATH_GEOMETRY_DELTA_BP = (0.006, 0.012)
PATH_MODEL_PREVIEW_BP = (0.003, 0.012)
PATH_TRACKING_ERROR_DEADZONE = 0.00025
PATH_TRACKING_EXTENSION_LIMIT = 0.004
PATH_MODEL_ERROR_DEADZONE = 0.0005
PATH_C0_CORRECTION_LIMIT = 0.02
PATH_C0_CONTINUATION_LIMIT = 0.04
PATH_C1_CORRECTION_LIMIT = 0.012
PATH_ARRIVAL_ERROR_BP = (0.0005, 0.002)
PATH_CONTINUATION_MARGIN = 0.006
PATH_C3_REALLOCATION_LOOKAHEAD = 15.0
PATH_DESIRED_TREND_SAMPLES = 6
PATH_DESIRED_RETREAT_DEADZONE_DEG = 2.0


@dataclass(frozen=True)
class Lmc2Polynomial:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


@dataclass(frozen=True)
class SteeringFeedback:
  measured_curvature: float
  projected_curvature: float
  desired_angle_curvature: float
  desired_angle_deg: float
  speed: float
  active: bool
  driver_override: bool
  lat_ctl_limit: int = 0


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _interp(value: float, lower: float, upper: float, lower_value: float, upper_value: float) -> float:
  if value <= lower:
    return lower_value
  if value >= upper:
    return upper_value
  return lower_value + (value - lower) / (upper - lower) * (upper_value - lower_value)


def _blend(first: float, second: float, second_share: float) -> float:
  return first + second_share * (second - first)


def _equivalent_curvature(coefficients: tuple[float, float, float, float],
                          distance: float = PATH_MIN_LOOKAHEAD) -> float:
  c0, c1, c2, c3 = coefficients
  return 2.0 * c0 / distance ** 2 + 2.0 * c1 / distance + c2 + c3 * distance / 3.0


def _tracking_error(target: float, measured: float, projected: float) -> float:
  measured_error = target - measured
  projected_error = target - projected
  if measured_error * target <= 0.0 or projected_error * target <= 0.0:
    return 0.0
  return math.copysign(min(abs(measured_error), abs(projected_error)), target)


def _add_near_curvature(coefficients: tuple[float, float, float, float],
                        curvature: float) -> tuple[float, float, float, float]:
  values = list(coefficients)
  remaining = curvature
  for index, basis in ((0, 2.0 / PATH_MIN_LOOKAHEAD ** 2), (1, 2.0 / PATH_MIN_LOOKAHEAD)):
    requested = values[index] + remaining / basis
    updated = _clip(requested, PATH_LIMITS[index])
    remaining -= (updated - values[index]) * basis
    values[index] = updated
  return tuple(values)


def _extend_while_behind(coefficients: tuple[float, float, float, float],
                         desired: float, measured: float,
                         projected: float) -> tuple[float, float, float, float]:
  wheel_error = _tracking_error(desired, measured, projected)
  command_error = desired - _equivalent_curvature(coefficients)
  if wheel_error * desired <= 0.0 or command_error * desired <= 0.0:
    return coefficients

  extension = min(
    max(min(abs(wheel_error), abs(command_error)) - PATH_TRACKING_ERROR_DEADZONE, 0.0),
    PATH_TRACKING_EXTENSION_LIMIT,
  )
  return _add_near_curvature(coefficients, math.copysign(extension, desired))


def _c3_compatibility_share(curvature_rate: float, desired: float,
                            measured: float, projected: float) -> float:
  if curvature_rate * desired >= 0.0:
    return 1.0
  tracking_error = _tracking_error(desired, measured, projected)
  return 1.0 - _interp(
    abs(tracking_error),
    *PATH_ARRIVAL_ERROR_BP,
    0.0,
    1.0,
  )


def _model_targets(coefficients: tuple[float, float, float, float],
                   speed: float) -> tuple[float, float, float]:
  c0, c1, c2, _ = coefficients
  lookahead = max(speed, PATH_MIN_LOOKAHEAD)
  offset_target = 2.0 * c0 / PATH_MIN_LOOKAHEAD ** 2
  angle_target = c1 / lookahead
  if offset_target * angle_target <= 0.0:
    return c2, c2, c2

  preview_share = _interp(
    max(abs(offset_target), abs(angle_target)),
    *PATH_MODEL_PREVIEW_BP,
    0.0,
    1.0,
  )
  geometry_target = math.copysign(
    min(abs(offset_target), abs(angle_target)),
    offset_target,
  )
  reference = c2 if geometry_target * c2 >= 0.0 else 0.0
  return (
    _blend(reference, offset_target, preview_share),
    _blend(reference, angle_target, preview_share),
    _blend(reference, geometry_target, preview_share),
  )


def _gated_model_correction(model_target: float, desired: float,
                            measured: float, projected: float,
                            limit: float) -> float:
  desired_error = _tracking_error(desired, measured, projected)
  if desired_error == 0.0:
    return 0.0

  model_error = model_target - measured
  correction = desired_error
  if model_error * desired > 0.0 and abs(model_error) > abs(correction):
    correction = model_error
  correction = math.copysign(
    min(max(abs(correction) - PATH_MODEL_ERROR_DEADZONE, 0.0), limit),
    desired,
  )
  arrival_share = _interp(
    abs(desired_error),
    *PATH_ARRIVAL_ERROR_BP,
    0.0,
    1.0,
  )
  return correction * arrival_share


def _reallocate_c3_nearer(coefficients: tuple[float, float, float, float],
                          model_target: float, desired: float,
                          measured: float, projected: float,
                          lat_ctl_limit: int) -> tuple[float, float, float, float]:
  envelope_share = 0.5 if lat_ctl_limit == 1 else 1.0 if lat_ctl_limit == 2 else 0.0
  c0, c1, c2, c3 = coefficients
  if envelope_share == 0.0 or c3 * desired <= 0.0 or c3 * model_target <= 0.0:
    return coefficients

  tracking_error = _tracking_error(desired, measured, projected)
  arrival_share = _interp(
    abs(tracking_error),
    *PATH_ARRIVAL_ERROR_BP,
    0.0,
    1.0,
  )
  requested_spill = c3 * envelope_share * arrival_share
  c0_with_spill = _clip(
    c0 + requested_spill * PATH_C3_REALLOCATION_LOOKAHEAD ** 3 / 6.0,
    PATH_LIMITS[0],
  )
  actual_spill = (c0_with_spill - c0) * 6.0 / PATH_C3_REALLOCATION_LOOKAHEAD ** 3
  return c0_with_spill, c1, c2, c3 - actual_spill


def _outward_beyond_error(target: float, wheel_curvature: float) -> float:
  if target == 0.0:
    return abs(wheel_curvature)
  direction = math.copysign(1.0, target)
  return max((wheel_curvature - target) * direction, 0.0)


def _terminal_guard(coefficients: tuple[float, float, float, float],
                    desired: float, measured: float, projected: float) \
                    -> tuple[float, float, float, float]:
  command_curvature = _equivalent_curvature(coefficients)
  if command_curvature * measured <= 0.0:
    return coefficients

  beyond_error = min(
    _outward_beyond_error(desired, measured),
    _outward_beyond_error(desired, projected),
  )
  arrival_share = _interp(beyond_error, *PATH_ARRIVAL_ERROR_BP, 0.0, 1.0)
  if arrival_share == 0.0:
    return coefficients

  corridor = abs(desired) + PATH_CONTINUATION_MARGIN
  c0, c1, c2, c3 = coefficients
  if c3 * measured < 0.0:
    direction = math.copysign(1.0, measured)
    near_curvature = direction * _equivalent_curvature((c0, c1, c2, 0.0))
    rate_curvature = direction * c3 * PATH_MIN_LOOKAHEAD / 3.0
    full_scale = _clip((corridor - rate_curvature) / near_curvature, (0.0, 1.0)) \
      if near_curvature > 0.0 else 1.0
    scale = _interp(arrival_share, 0.0, 1.0, 1.0, full_scale)
    return c0 * scale, c1 * scale, c2 * scale, c3

  full_scale = min(abs(command_curvature), corridor) / abs(command_curvature)
  scale = _interp(arrival_share, 0.0, 1.0, 1.0, full_scale)
  return tuple(value * scale for value in coefficients)


def _maneuver_share(coefficients: tuple[float, float, float, float], speed: float, valid: bool) -> float:
  c0, c1, c2, c3 = coefficients
  lookahead = max(speed, PATH_MIN_LOOKAHEAD)
  if not valid:
    return _interp(
      max(abs(c2) - PATH_LIMITS[2][1], 0.0),
      *PATH_C2_BASEBAND_BP,
      0.0,
      1.0,
    )

  offset_curvature = 2.0 * c0 / PATH_MIN_LOOKAHEAD ** 2
  angle_curvature = c1 / lookahead
  geometry_delta = max(
    abs(offset_curvature - c2),
    abs(angle_curvature - c2),
  ) if offset_curvature * angle_curvature > 0.0 else 0.0
  return max(
    _interp(abs(c3) * lookahead / 3.0, *PATH_C2_BASEBAND_BP, 0.0, 1.0),
    _interp(geometry_delta, *PATH_GEOMETRY_DELTA_BP, 0.0, 1.0),
    _interp(max(abs(c2) - PATH_LIMITS[2][1], 0.0), *PATH_C2_BASEBAND_BP, 0.0, 1.0),
  )


class FordPolynomialServo:
  """Convert model intent and steering feedback into one bounded polynomial."""

  def __init__(self):
    self._desired_angles: deque[float] = deque(maxlen=PATH_DESIRED_TREND_SAMPLES)
    self._desired_direction = 0.0
    self._terminal_envelope = False

  def _reset(self) -> None:
    self._desired_angles.clear()
    self._desired_direction = 0.0
    self._terminal_envelope = False

  def _desired_is_retreating(self, desired_curvature: float, desired_angle_deg: float) -> bool:
    if desired_curvature * self._desired_direction < 0.0:
      self._reset()
    if desired_curvature != 0.0:
      self._desired_direction = math.copysign(1.0, desired_curvature)

    self._desired_angles.append(abs(desired_angle_deg))
    if len(self._desired_angles) < self._desired_angles.maxlen:
      return False

    angles = tuple(self._desired_angles)
    recent_retreat = angles[-1] <= angles[-2] <= angles[-3]
    return recent_retreat and max(angles[:-1]) - angles[-1] > PATH_DESIRED_RETREAT_DEADZONE_DEG

  def update(self, path, feedback: SteeringFeedback) -> Lmc2Polynomial:
    if not feedback.active:
      self._reset()
      return Lmc2Polynomial()

    valid = path is not None and bool(getattr(path, "valid", False))
    if feedback.driver_override:
      self._reset()
      measured_curvature = _finite(feedback.measured_curvature)
      lookahead = max(_finite(feedback.speed), PATH_MIN_LOOKAHEAD)
      return Lmc2Polynomial(
        valid=valid,
        path_offset=_clip(0.5 * measured_curvature * PATH_MIN_LOOKAHEAD ** 2, PATH_LIMITS[0]),
        path_angle=_clip(measured_curvature * lookahead, PATH_LIMITS[1]),
      )

    coefficients = (
      _finite(getattr(path, "pathOffset", 0.0)) if valid else 0.0,
      _finite(getattr(path, "pathAngle", 0.0)) if valid else 0.0,
      _finite(getattr(path, "curvature", 0.0)) if path is not None else 0.0,
      _finite(getattr(path, "curvatureRate", 0.0)) if valid else 0.0,
    )
    if not valid:
      self._reset()
      return Lmc2Polynomial(curvature=_clip(coefficients[2], PATH_LIMITS[2]))

    measured_curvature = _finite(feedback.measured_curvature)
    projected_curvature = _finite(feedback.projected_curvature, measured_curvature)
    desired_curvature = _finite(feedback.desired_angle_curvature, coefficients[2])
    speed = max(_finite(feedback.speed), 0.0)
    lookahead = max(speed, PATH_MIN_LOOKAHEAD)
    maneuver_share = _maneuver_share(coefficients, speed, valid)
    desired_retreating = self._desired_is_retreating(
      desired_curvature,
      _finite(feedback.desired_angle_deg),
    )
    model_unwinding = maneuver_share > 0.0 and coefficients[3] * measured_curvature < 0.0
    self._terminal_envelope |= desired_retreating or model_unwinding
    c3_share = _c3_compatibility_share(
      coefficients[3],
      desired_curvature,
      measured_curvature,
      projected_curvature,
    )
    offset_target, angle_target, model_target = _model_targets(coefficients, speed)
    c3_limit = PATH_LIMITS[3][1] if coefficients[3] >= 0.0 else abs(PATH_LIMITS[3][0])
    outward_c3_is_pinned = coefficients[3] * measured_curvature > 0.0 and \
                           abs(coefficients[3]) >= c3_limit
    continuing_preview = coefficients[3] * model_target > 0.0 and \
                         abs(coefficients[3]) >= c3_limit and \
                         abs(model_target) > abs(desired_curvature) + PATH_MODEL_ERROR_DEADZONE
    c0_limit = PATH_C0_CONTINUATION_LIMIT if continuing_preview else PATH_C0_CORRECTION_LIMIT
    offset_target += _gated_model_correction(
      offset_target,
      desired_curvature,
      measured_curvature,
      projected_curvature,
      c0_limit,
    )
    angle_target += _gated_model_correction(
      angle_target,
      desired_curvature,
      measured_curvature,
      projected_curvature,
      PATH_C1_CORRECTION_LIMIT,
    )
    command = (
      0.5 * offset_target * PATH_MIN_LOOKAHEAD ** 2 * maneuver_share,
      angle_target * lookahead * maneuver_share,
      coefficients[2] * (1.0 - maneuver_share),
      coefficients[3] * c3_share * maneuver_share,
    )
    command = tuple(_clip(value, limits) for value, limits in zip(command, PATH_LIMITS, strict=True))
    if _equivalent_curvature(coefficients) * desired_curvature > 0.0:
      command = _extend_while_behind(
        command,
        desired_curvature,
        measured_curvature,
        projected_curvature,
      )
    command = _reallocate_c3_nearer(
      command,
      model_target,
      desired_curvature,
      measured_curvature,
      projected_curvature,
      int(feedback.lat_ctl_limit),
    )
    if self._terminal_envelope or outward_c3_is_pinned:
      command = _terminal_guard(
        command,
        desired_curvature,
        measured_curvature,
        projected_curvature,
      )

    return Lmc2Polynomial(
      valid=valid,
      path_offset=command[0],
      path_angle=command[1],
      curvature=command[2],
      curvature_rate=command[3],
    )
