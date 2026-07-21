"""Model-aware LMC2 candidate used only by the isolated shadow logger."""

from dataclasses import dataclass
import math


PATH_C0_LIMITS = (-4.61, 4.60)
PATH_C1_LIMITS = (-0.475, 0.497)
PATH_C2_LIMITS = (-0.02, 0.02)
PATH_C3_LIMITS = (-0.001024, 0.001023)
PATH_C2_FADE_BP = (0.006, 0.012)
PATH_C2_SLEW = 0.0002
PATH_C2_LATCH_ENTER = PATH_C2_FADE_BP[1]
PATH_C2_LATCH_EXIT_TARGET = 0.002
PATH_C2_LATCH_EXIT_ERROR = 0.001
PATH_C2_RECOVERY_FRAMES = 10
PATH_MANEUVER_CURVATURE_SLEW = 0.006
PATH_C3_SLEW = 0.0002
PATH_C0_DISTANCE = 7.0
PATH_MIN_LOOKAHEAD = 7.0
PATH_C3_HORIZONS = (3.5, 5.0, 7.0)


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


def _limit_same_direction_attack(value: float, last: float, max_step: float) -> float:
  if value * last >= 0.0 and abs(value) > abs(last):
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


class LatControlPath:
  """Convert model path intent into one bounded local polynomial command."""

  def __init__(self):
    self._last_curvature = 0.0
    self._last_command = LateralPathCommand()
    self._c2_latched = False
    self._c2_recovery_frames = 0

  def update(self, model, desired_curvature: float, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool) -> LateralPathCommand:
    desired_curvature = _finite(desired_curvature)
    measured_curvature = _finite(measured_curvature)
    v_ego = max(_finite(v_ego), 0.0)
    if not active:
      if self._c2_latched:
        self._c2_recovery_frames += 1
        if self._c2_recovery_frames >= PATH_C2_RECOVERY_FRAMES:
          self._c2_latched = False
          self._c2_recovery_frames = 0
      self._last_curvature = 0.0
      self._last_command = LateralPathCommand()
      return LateralPathCommand()

    lookahead = max(v_ego, PATH_MIN_LOOKAHEAD)
    if driver_override:
      self._last_curvature = 0.0
      self._last_command = LateralPathCommand(
        valid=True,
        path_offset=_clip(0.5 * measured_curvature * PATH_C0_DISTANCE ** 2, PATH_C0_LIMITS),
        path_angle=_clip(measured_curvature * lookahead, PATH_C1_LIMITS),
      )
      return self._last_command

    path = _model_path(model)
    if path is None:
      path_offset = 0.0
      path_angle = 0.0
      spatial_curvature_rate = 0.0
    else:
      distances, offsets, headings = path
      path_offset = _sample(PATH_C0_DISTANCE, distances, offsets)
      path_angle = _sample(lookahead, distances, headings)
      spatial_curvature_rate = _curvature_rate(path)

    if abs(desired_curvature) >= PATH_C2_LATCH_ENTER:
      self._c2_latched = True
      self._c2_recovery_frames = 0
    elif self._c2_latched:
      near_target = abs(desired_curvature) <= PATH_C2_LATCH_EXIT_TARGET
      tracking = abs(desired_curvature - measured_curvature) <= PATH_C2_LATCH_EXIT_ERROR
      if near_target and tracking:
        self._c2_recovery_frames += 1
        if self._c2_recovery_frames >= PATH_C2_RECOVERY_FRAMES:
          self._c2_latched = False
          self._c2_recovery_frames = 0
      else:
        self._c2_recovery_frames = 0

    instantaneous_c2_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 1.0, 0.0)
    c2_share = 0.0 if self._c2_latched else instantaneous_c2_share
    target_curvature = _clip(desired_curvature * c2_share, PATH_C2_LIMITS)
    if target_curvature * self._last_curvature >= 0.0 and abs(target_curvature) > abs(self._last_curvature):
      target_curvature = math.copysign(
        min(abs(target_curvature), abs(self._last_curvature) + PATH_C2_SLEW),
        target_curvature,
      )
    elif target_curvature * self._last_curvature < 0.0:
      target_curvature = 0.0
    self._last_curvature = target_curvature

    path_curvature = path_angle / lookahead
    offset_curvature = 2.0 * path_offset / (PATH_C0_DISTANCE ** 2)
    delivered_curvature = 0.0
    if measured_curvature * path_curvature > 0.0:
      delivered_curvature = math.copysign(
        min(abs(measured_curvature), abs(path_curvature)), path_curvature,
      )

    c2_path_share = c2_share
    if path_curvature != 0.0 and path_curvature * desired_curvature > 0.0:
      c2_authority = abs(target_curvature)
      if c2_share == 1.0:
        c2_authority = max(c2_authority, abs(desired_curvature))
      c2_path_share = min(c2_authority / abs(path_curvature), 1.0)
    delivered_c2_curvature = delivered_curvature * c2_path_share

    maneuver_curvature = max(abs(desired_curvature), abs(path_curvature), abs(offset_curvature))
    c0_share = _interp(maneuver_curvature, 0.003, 0.006, 0.0, 1.0)
    live_c3_share = _interp(abs(desired_curvature), *PATH_C2_FADE_BP, 0.0, 1.0)
    coherent_reversal = measured_curvature * path_curvature < 0.0 and \
                        spatial_curvature_rate * path_curvature > 0.0 and \
                        offset_curvature * path_curvature > 0.0
    preview_c3_share = _interp(maneuver_curvature, *PATH_C2_FADE_BP, 0.0, 1.0) if coherent_reversal else 0.0
    c3_share = max(live_c3_share, preview_c3_share)

    path_offset_command = _clip(
      (path_offset - 0.5 * delivered_c2_curvature * PATH_C0_DISTANCE ** 2) * c0_share,
      PATH_C0_LIMITS,
    )
    path_angle_command = _clip(path_angle - delivered_c2_curvature * lookahead, PATH_C1_LIMITS)
    curvature_rate_command = _clip(spatial_curvature_rate * c3_share, PATH_C3_LIMITS)
    path_offset_command = _limit_same_direction_attack(
      path_offset_command,
      self._last_command.path_offset,
      0.5 * PATH_MANEUVER_CURVATURE_SLEW * PATH_C0_DISTANCE ** 2,
    )
    path_angle_command = _limit_same_direction_attack(
      path_angle_command,
      self._last_command.path_angle,
      PATH_MANEUVER_CURVATURE_SLEW * lookahead,
    )
    curvature_rate_command = _limit_same_direction_attack(
      curvature_rate_command,
      self._last_command.curvature_rate,
      PATH_C3_SLEW,
    )

    self._last_command = LateralPathCommand(
      valid=path is not None,
      path_offset=path_offset_command,
      path_angle=path_angle_command,
      curvature=target_curvature,
      curvature_rate=curvature_rate_command,
    )
    return self._last_command
