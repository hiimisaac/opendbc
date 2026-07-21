"""Action-tracking LMC2 candidate used only by the isolated shadow logger."""

from dataclasses import dataclass
import math


PATH_C0_LIMITS = (-4.61, 4.60)
PATH_C1_LIMITS = (-0.475, 0.497)
PATH_C2_LIMITS = (-0.02, 0.02)
PATH_C3_LIMITS = (-0.001024, 0.001023)

PATH_C0_DISTANCE = 7.0
PATH_MIN_LOOKAHEAD = 7.0
PATH_C3_HORIZONS = (3.5, 5.0, 7.0)

PATH_C2_FADE_BP = (0.006, 0.012)
PATH_C2_SETTLED_BP = (0.003, 0.006)
PATH_C2_SLEW = 0.0002
PATH_PREVIEW_ERROR_BP = (0.0005, 0.003)
PATH_PREVIEW_ACTION_BP = (0.003, 0.006)
PATH_PREVIEW_COHERENCE_BP = (0.2, 0.6)
PATH_UNWIND_ERROR_DEADZONE = 0.0005
PATH_UNWIND_LIMIT = 0.006
PATH_MANEUVER_CURVATURE_SLEW = 0.006
PATH_C3_SLEW = 0.0002


@dataclass(frozen=True)
class LateralPathCommand:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


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


def _limit_attack(value: float, last: float, max_step: float) -> float:
  """Bound authority growth, release immediately, and reverse without a jump."""
  if value * last < 0.0:
    return math.copysign(min(abs(value), max_step), value)
  if abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _sample(distance: float, distances: list[float], values: list[float]) -> float:
  if distance <= distances[0]:
    return values[0]
  if distance >= distances[-1]:
    return values[-1]
  for i in range(1, len(distances)):
    if distance <= distances[i]:
      span = distances[i] - distances[i - 1]
      alpha = 0.0 if span == 0.0 else (distance - distances[i - 1]) / span
      return values[i - 1] + alpha * (values[i] - values[i - 1])
  return values[-1]


def _model_path(model) -> tuple[list[float], list[float], list[float]] | None:
  if model is None:
    return None
  try:
    xs = [float(value) for value in model.position.x]
    ys = [float(value) for value in model.position.y]
    headings = [float(value) for value in model.orientation.z]
  except (AttributeError, TypeError, ValueError):
    return None
  if len(xs) < 2 or len(xs) != len(ys) or len(xs) != len(headings):
    return None
  if not all(math.isfinite(value) for values in (xs, ys, headings) for value in values):
    return None

  distances = [0.0]
  for i in range(1, len(xs)):
    distances.append(distances[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
  if distances[-1] <= 0.0:
    return None

  unwrapped_headings = [headings[0]]
  for heading in headings[1:]:
    delta = (heading - unwrapped_headings[-1] + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped_headings.append(unwrapped_headings[-1] + delta)
  return distances, ys, unwrapped_headings


def _curvature_rate(path: tuple[list[float], list[float], list[float]]) -> float:
  distances, _, headings = path
  rates = []
  for requested_horizon in PATH_C3_HORIZONS:
    horizon = min(requested_horizon, distances[-1])
    if horizon <= 0.0:
      continue
    start = _sample(0.0, distances, headings)
    midpoint = _sample(0.5 * horizon, distances, headings)
    end = _sample(horizon, distances, headings)
    rates.append(4.0 * (start - 2.0 * midpoint + end) / (horizon * horizon))
  if not rates:
    return 0.0
  while len(rates) < 3:
    rates.append(rates[-1])
  magnitude = sum(abs(rate) for rate in rates)
  if magnitude == 0.0:
    return 0.0
  return sorted(rates)[1] * abs(sum(rates)) / magnitude


def _preview_target(path_curvature: float, desired_curvature: float,
                    tracking_error: float, wheel_beyond_target: bool) -> float:
  """Return path preview bounded by action support and measured tracking."""
  preview_excess = path_curvature - desired_curvature
  action_share = 0.0 if wheel_beyond_target else \
    _interp(abs(desired_curvature), *PATH_PREVIEW_ACTION_BP, 0.0, 1.0)
  coherence = 0.0
  if not wheel_beyond_target and \
      path_curvature * desired_curvature > 0.0 and path_curvature != 0.0:
    coherence = min(abs(desired_curvature / path_curvature), 1.0)
  coherence_share = _interp(coherence, *PATH_PREVIEW_COHERENCE_BP, 0.0, 1.0)
  preview_share = max(action_share, coherence_share)
  if preview_excess * tracking_error > 0.0:
    error_share = _interp(abs(tracking_error), *PATH_PREVIEW_ERROR_BP, 0.0, 1.0)
    preview_share = max(preview_share, error_share)
  return desired_curvature + preview_share * preview_excess


class LatControlPath:
  """Map action, path preview, and measured wheel curvature to one polynomial.

  The previous command is the only persistent controller state. The action head
  owns the current target; path geometry can add authority while it closes the
  measured tracking error or the action still supports the upcoming maneuver.
  """

  def __init__(self):
    self._last_command = LateralPathCommand()

  def update(self, model, desired_curvature: float, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool) -> LateralPathCommand:
    desired_curvature = _finite(desired_curvature)
    measured_curvature = _finite(measured_curvature)
    v_ego = max(_finite(v_ego), 0.0)

    if not active:
      self._last_command = LateralPathCommand()
      return self._last_command

    lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
    path = _model_path(model)
    valid = path is not None
    if path is None:
      path_offset = 0.0
      path_angle = 0.0
      spatial_curvature_rate = 0.0
    else:
      distances, offsets, headings = path
      path_offset = _sample(PATH_C0_DISTANCE, distances, offsets)
      path_angle = _sample(lookahead, distances, headings)
      spatial_curvature_rate = _curvature_rate(path)

    if driver_override:
      command = LateralPathCommand(
        valid=valid,
        path_offset=_clip(0.5 * measured_curvature * PATH_C0_DISTANCE ** 2, PATH_C0_LIMITS),
        path_angle=_clip(measured_curvature * lookahead, PATH_C1_LIMITS),
      )
      self._last_command = command
      return command

    tracking_error = desired_curvature - measured_curvature
    wheel_beyond_target = tracking_error * measured_curvature < 0.0
    offset_curvature = 2.0 * path_offset / PATH_C0_DISTANCE ** 2 if valid else 0.0
    angle_curvature = path_angle / lookahead if valid else 0.0

    if valid:
      offset_target = _preview_target(
        offset_curvature, desired_curvature, tracking_error, wheel_beyond_target,
      )
      angle_target = _preview_target(
        angle_curvature, desired_curvature, tracking_error, wheel_beyond_target,
      )
    else:
      offset_target = 0.0
      angle_target = 0.0

    # Once the wheel is beyond the action target, actively remove the old turn.
    # This correction cannot relatch stale preview because it is applied only in
    # the direction opposite measured wheel curvature.
    unwind = 0.0
    if wheel_beyond_target:
      unwind = _clip(
        _deadzone(tracking_error, PATH_UNWIND_ERROR_DEADZONE),
        (-PATH_UNWIND_LIMIT, PATH_UNWIND_LIMIT),
      )
      offset_target += unwind
      angle_target += unwind

    c2_action_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 1.0, 0.0)
    unresolved = max(abs(desired_curvature), abs(measured_curvature), abs(tracking_error))
    c2_settled_share = _interp(unresolved, *PATH_C2_SETTLED_BP, 1.0, 0.0)
    c2_share = min(c2_action_share, c2_settled_share)
    allocated_c2 = _clip(desired_curvature * c2_share, PATH_C2_LIMITS)
    curvature = allocated_c2
    curvature = _limit_attack(curvature, self._last_command.curvature, PATH_C2_SLEW)

    maneuver_demand = max(abs(desired_curvature), abs(offset_curvature), abs(angle_curvature))
    c0_share = _interp(maneuver_demand, 0.003, 0.006, 0.0, 1.0)
    if valid:
      path_offset_command = _clip(
        0.5 * (offset_target - allocated_c2) * PATH_C0_DISTANCE ** 2 * c0_share,
        PATH_C0_LIMITS,
      )
      path_angle_command = _clip((angle_target - allocated_c2) * lookahead, PATH_C1_LIMITS)
    else:
      path_offset_command = 0.0
      path_angle_command = 0.0

    c3_action_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 0.0, 1.0)
    coherent_reversal = measured_curvature * angle_curvature < 0.0 and \
                        spatial_curvature_rate * angle_curvature > 0.0 and \
                        offset_curvature * angle_curvature > 0.0
    c3_preview_share = _interp(maneuver_demand, *PATH_C2_FADE_BP, 0.0, 1.0) if coherent_reversal else 0.0
    c3_share = max(c3_action_share, c3_preview_share)
    curvature_rate_command = _clip(spatial_curvature_rate * c3_share, PATH_C3_LIMITS)

    path_offset_command = _limit_attack(
      path_offset_command,
      self._last_command.path_offset,
      0.5 * PATH_MANEUVER_CURVATURE_SLEW * PATH_C0_DISTANCE ** 2,
    )
    path_angle_command = _limit_attack(
      path_angle_command,
      self._last_command.path_angle,
      PATH_MANEUVER_CURVATURE_SLEW * lookahead,
    )
    curvature_rate_command = _limit_attack(
      curvature_rate_command,
      self._last_command.curvature_rate,
      PATH_C3_SLEW,
    )

    command = LateralPathCommand(
      valid=valid,
      path_offset=path_offset_command,
      path_angle=path_angle_command,
      curvature=curvature,
      curvature_rate=curvature_rate_command,
    )
    self._last_command = command
    return command
