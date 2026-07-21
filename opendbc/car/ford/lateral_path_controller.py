from dataclasses import dataclass
import math

from opendbc.car.ford.lateral_path_state import projected_tracking_error


# LMC2 describes one local cubic path with offset (c0), angle (c1), curvature
# (c2), and spatial curvature slope (c3). The model provides the geometry while
# measured steering closes only bounded transient error. Persistent state is
# deliberately limited to turn commitment, c2 readiness, and the last command.
FORD_PATH_C0_CAN_CLIP = (-4.61, 4.60)
FORD_PATH_C1_CAN_CLIP = (-0.475, 0.497)
FORD_PATH_C2_CAN_CLIP = (-0.02, 0.02)
FORD_PATH_C3_CAN_CLIP = (-0.001024, 0.001023)

FORD_PATH_D_C0 = 7.0
FORD_PATH_D_LOOK_MIN = 7.0
FORD_PATH_D_LOOK_TIME = 1.0
FORD_PATH_PREVIEW_TIME = 0.5
FORD_PATH_PREVIEW_MAX = 12.0
FORD_PATH_C3_HORIZONS = (3.5, 5.0, 7.0)

FORD_PATH_MANEUVER_BP = (0.003, 0.006)
FORD_PATH_COMMITMENT_RELEASE_STEP = 0.1
FORD_PATH_C1_DEADZONE = 0.0003
FORD_PATH_C1_QUIET_DEADZONE = 0.001

FORD_PATH_C2_FADE_BP = (0.006, 0.012)
FORD_PATH_C2_LATCH_ENTER = FORD_PATH_C2_FADE_BP[1]
FORD_PATH_C2_SETTLED_TARGET = 0.002
FORD_PATH_C2_SETTLED_WHEEL = 0.003
FORD_PATH_C2_SETTLED_ERROR = 0.001
FORD_PATH_C2_READINESS_STEP = 0.1
FORD_PATH_C2_SLEW = 0.0002

FORD_PATH_FEEDBACK_RATIO = 0.2
FORD_PATH_FEEDBACK_LIMIT = 0.02
FORD_PATH_UNWIND_LIMIT = 0.006
FORD_PATH_UNWIND_ANGLE_DEADZONE_DEG = 2.5
FORD_PATH_C3_ERROR_DEADZONE = 0.0005

FORD_PATH_MANEUVER_ATTACK_SLEW = 0.006
FORD_PATH_HANDOFF_SLEW = 0.01
FORD_PATH_C3_ATTACK_SLEW = 0.0002


@dataclass(frozen=True)
class LateralPathCommand:
  curvature: float = 0.0
  curvature_rate: float = 0.0
  path_angle: float = 0.0
  path_offset: float = 0.0


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


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


def _limit_same_direction_attack(value: float, last: float, max_step: float) -> float:
  if value * last >= 0.0 and abs(value) > abs(last):
    return math.copysign(min(abs(value), abs(last) + max_step), value)
  return value


def _limit_step(value: float, last: float, max_step: float) -> float:
  return min(max(value, last - max_step), last + max_step)


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


def model_curvature_rate(model, horizon: float) -> float:
  path = _model_path(model)
  if path is None:
    return 0.0
  distances, _, headings = path
  horizon = min(max(_finite(horizon), 0.0), distances[-1])
  if horizon <= 0.0:
    return 0.0
  start = _sample(0.0, distances, headings)
  midpoint = _sample(0.5 * horizon, distances, headings)
  end = _sample(horizon, distances, headings)
  return _finite(4.0 * (start - 2.0 * midpoint + end) / (horizon * horizon))


def model_curvature_rate_consensus(model) -> float:
  rates = [model_curvature_rate(model, horizon) for horizon in FORD_PATH_C3_HORIZONS]
  magnitude = sum(abs(rate) for rate in rates)
  if magnitude == 0.0:
    return 0.0
  return _finite(sorted(rates)[1] * abs(sum(rates)) / magnitude)


def _model_target(c0_curvature: float, c1_curvature: float, desired_curvature: float) -> float:
  if c0_curvature * c1_curvature <= 0.0:
    return desired_curvature
  target = 0.5 * (c0_curvature + c1_curvature)
  if target * desired_curvature > 0.0 and abs(target) < abs(desired_curvature):
    return desired_curvature
  return target


def _authority_floor(path_value: float, requested_value: float) -> float:
  if path_value * requested_value > 0.0 and abs(path_value) >= abs(requested_value):
    return path_value
  return requested_value


def _same_direction_cap(path_value: float, limit_value: float) -> float:
  if path_value * limit_value > 0.0 and abs(path_value) > abs(limit_value):
    return limit_value
  return path_value


class LateralPathController:
  """Convert model geometry and measured tracking into one LMC2 polynomial."""

  def __init__(self):
    self._turn_commitment = 0.0
    self._c2_readiness = 1.0
    self._last_command = LateralPathCommand()
    self._handoff_active = False

  @property
  def handoff_active(self) -> bool:
    return self._handoff_active

  def _update_turn_commitment(self, demand: float) -> None:
    target = _interp(abs(demand), *FORD_PATH_MANEUVER_BP, 0.0, 1.0)
    if target >= self._turn_commitment:
      self._turn_commitment = target
    else:
      self._turn_commitment = max(target, self._turn_commitment - FORD_PATH_COMMITMENT_RELEASE_STEP)

  def _update_c2_readiness(self, demand: float, desired_curvature: float, wheel_curvature: float, angle_error_curvature: float) -> None:
    if abs(demand) >= FORD_PATH_C2_LATCH_ENTER:
      self._c2_readiness = 0.0
      return

    settled = (
      abs(desired_curvature) <= FORD_PATH_C2_SETTLED_TARGET
      and abs(wheel_curvature) <= FORD_PATH_C2_SETTLED_WHEEL
      and abs(angle_error_curvature) <= FORD_PATH_C2_SETTLED_ERROR
    )
    if settled:
      self._c2_readiness = min(self._c2_readiness + FORD_PATH_C2_READINESS_STEP, 1.0)

  def update(
    self,
    model,
    desired_curvature: float,
    k_meas: float,
    v_ego: float,
    lat_active: bool,
    driver_override: bool,
    *,
    angle_error_curvature: float = 0.0,
    wheel_curvature: float = 0.0,
    projected_wheel_curvature: float | None = None,
    hold_command: bool = False,
  ) -> LateralPathCommand:
    desired_curvature = _finite(desired_curvature)
    k_meas = _finite(k_meas)
    v_ego = max(_finite(v_ego), 0.0)
    angle_error_curvature = _finite(angle_error_curvature)
    wheel_curvature = _finite(wheel_curvature, k_meas)
    feedback_available = projected_wheel_curvature is not None
    projected_wheel_curvature = _finite(projected_wheel_curvature, wheel_curvature) if feedback_available else wheel_curvature

    if not lat_active:
      self._turn_commitment = 0.0
      self._update_c2_readiness(0.0, 0.0, 0.0, 0.0)
      self._last_command = LateralPathCommand()
      self._handoff_active = False
      return self._last_command

    d_look = max(v_ego * FORD_PATH_D_LOOK_TIME, FORD_PATH_D_LOOK_MIN)
    path = _model_path(model)
    if path is None:
      raw_path_offset = 0.5 * desired_curvature * FORD_PATH_D_C0**2
      raw_path_angle = desired_curvature * d_look
      curvature_rate = 0.0
    else:
      distances, offsets, headings = path
      raw_path_offset = _sample(FORD_PATH_D_C0, distances, offsets)
      raw_path_angle = _sample(d_look, distances, headings)
      curvature_rate = model_curvature_rate_consensus(model)

      preview_distance = min(
        d_look + v_ego * FORD_PATH_PREVIEW_TIME,
        FORD_PATH_PREVIEW_MAX,
        distances[-1],
      )
      if preview_distance > d_look and angle_error_curvature * wheel_curvature >= 0.0:
        near_c0_curvature = 2.0 * raw_path_offset / FORD_PATH_D_C0**2
        near_c1_curvature = raw_path_angle / d_look
        preview_offset = _sample(preview_distance, distances, offsets)
        preview_angle = _sample(preview_distance, distances, headings)
        preview_c0_curvature = 2.0 * preview_offset / preview_distance**2
        preview_c1_curvature = preview_angle / preview_distance
        preview_strength = min(abs(preview_c0_curvature), abs(preview_c1_curvature))
        preview_share = _interp(preview_strength, *FORD_PATH_MANEUVER_BP, 0.0, 1.0)
        preview_agrees = preview_c0_curvature * preview_c1_curvature > 0.0 and curvature_rate * preview_c1_curvature > 0.0
        preview_builds = abs(preview_c0_curvature) > abs(near_c0_curvature) and abs(preview_c1_curvature) > abs(near_c1_curvature)
        if preview_agrees and preview_builds:
          near_c0_curvature += preview_share * (preview_c0_curvature - near_c0_curvature)
          near_c1_curvature += preview_share * (preview_c1_curvature - near_c1_curvature)
          raw_path_offset = 0.5 * near_c0_curvature * FORD_PATH_D_C0**2
          raw_path_angle = near_c1_curvature * d_look

    instantaneous_c2_share = _interp(abs(desired_curvature), *FORD_PATH_C2_FADE_BP, 1.0, 0.0)
    if instantaneous_c2_share < 1.0:
      raw_path_offset = _authority_floor(
        raw_path_offset,
        0.5 * desired_curvature * FORD_PATH_D_C0**2,
      )
      raw_path_angle = _authority_floor(raw_path_angle, desired_curvature * d_look)

    upstream_is_unwinding = (
      abs(desired_curvature) <= FORD_PATH_MANEUVER_BP[0]
      and abs(wheel_curvature) >= FORD_PATH_C2_LATCH_ENTER
      and desired_curvature * wheel_curvature >= 0.0
      and angle_error_curvature * wheel_curvature < 0.0
    )
    if upstream_is_unwinding:
      raw_path_offset = _same_direction_cap(
        raw_path_offset,
        0.5 * wheel_curvature * FORD_PATH_D_C0**2,
      )
      raw_path_angle = _same_direction_cap(
        raw_path_angle,
        wheel_curvature * d_look,
      )

    c0_curvature = 2.0 * raw_path_offset / FORD_PATH_D_C0**2
    c1_curvature = raw_path_angle / d_look
    model_target = _model_target(c0_curvature, c1_curvature, desired_curvature)
    maneuver_demand = math.copysign(
      max(abs(desired_curvature), abs(c0_curvature), abs(c1_curvature)),
      model_target if model_target != 0.0 else desired_curvature,
    )
    self._update_turn_commitment(maneuver_demand)
    self._update_c2_readiness(maneuver_demand, desired_curvature, wheel_curvature, angle_error_curvature)

    if driver_override:
      command = LateralPathCommand(
        path_angle=_clip(wheel_curvature * d_look, FORD_PATH_C1_CAN_CLIP),
        path_offset=_clip(0.5 * wheel_curvature * FORD_PATH_D_C0**2, FORD_PATH_C0_CAN_CLIP),
      )
      self._last_command = command
      self._handoff_active = True
      return command

    if hold_command:
      return self._last_command

    c2_ready = self._c2_readiness >= 1.0 - 1e-9
    c2_share = instantaneous_c2_share * float(c2_ready)
    allocated_c2 = _clip(desired_curvature * c2_share, FORD_PATH_C2_CAN_CLIP)
    curvature = allocated_c2
    if curvature * self._last_command.curvature < 0.0:
      curvature = 0.0
    else:
      curvature = _limit_same_direction_attack(
        curvature,
        self._last_command.curvature,
        FORD_PATH_C2_SLEW,
      )

    c1_deadzone = _interp(self._turn_commitment, 0.0, 1.0, FORD_PATH_C1_QUIET_DEADZONE, FORD_PATH_C1_DEADZONE)
    path_angle = _deadzone(c1_curvature - allocated_c2, c1_deadzone) * d_look
    path_offset = (raw_path_offset - 0.5 * allocated_c2 * FORD_PATH_D_C0**2) * self._turn_commitment

    unwind = 0.0
    if angle_error_curvature * wheel_curvature < 0.0 and self._turn_commitment > 0.0:
      unwind = _clip(angle_error_curvature, (-FORD_PATH_UNWIND_LIMIT, FORD_PATH_UNWIND_LIMIT)) * self._turn_commitment
      path_angle += unwind * d_look
      path_offset += 0.5 * unwind * FORD_PATH_D_C0**2
    elif feedback_available:
      feedback_error = projected_tracking_error(
        model_target,
        wheel_curvature,
        projected_wheel_curvature,
      )
      feedback_limit = min(FORD_PATH_FEEDBACK_LIMIT, FORD_PATH_FEEDBACK_RATIO * abs(model_target))
      feedback_error = _clip(feedback_error, (-feedback_limit, feedback_limit))
      if feedback_error * model_target > 0.0:
        path_offset += 0.5 * feedback_error * FORD_PATH_D_C0**2

    live_c3_share = _interp(abs(desired_curvature), *FORD_PATH_C2_FADE_BP, 0.0, 1.0)
    coherent_reversal = projected_wheel_curvature * model_target < 0.0 and curvature_rate * model_target > 0.0
    preview_c3_share = _interp(abs(model_target), *FORD_PATH_C2_FADE_BP, 0.0, 1.0) if coherent_reversal else 0.0
    c3_error = model_target - projected_wheel_curvature
    if abs(c3_error) <= FORD_PATH_C3_ERROR_DEADZONE or curvature_rate * c3_error <= 0.0:
      curvature_rate = 0.0
    else:
      curvature_rate *= self._turn_commitment * max(live_c3_share, preview_c3_share)
    curvature_rate = _clip(curvature_rate, FORD_PATH_C3_CAN_CLIP)

    path_angle = _clip(path_angle, FORD_PATH_C1_CAN_CLIP)
    path_offset = _clip(path_offset, FORD_PATH_C0_CAN_CLIP)
    curvature_rate = _limit_same_direction_attack(
      curvature_rate,
      self._last_command.curvature_rate,
      FORD_PATH_C3_ATTACK_SLEW,
    )

    if self._handoff_active:
      target_angle = path_angle
      target_offset = path_offset
      path_angle = _limit_step(
        target_angle,
        self._last_command.path_angle,
        FORD_PATH_HANDOFF_SLEW * d_look,
      )
      path_offset = _limit_step(
        target_offset,
        self._last_command.path_offset,
        0.5 * FORD_PATH_HANDOFF_SLEW * FORD_PATH_D_C0**2,
      )
      self._handoff_active = path_angle != target_angle or path_offset != target_offset
    else:
      path_angle = _limit_same_direction_attack(
        path_angle,
        self._last_command.path_angle,
        FORD_PATH_MANEUVER_ATTACK_SLEW * d_look,
      )
      path_offset = _limit_same_direction_attack(
        path_offset,
        self._last_command.path_offset,
        0.5 * FORD_PATH_MANEUVER_ATTACK_SLEW * FORD_PATH_D_C0**2,
      )

    command = LateralPathCommand(
      curvature=curvature,
      curvature_rate=curvature_rate,
      path_angle=path_angle,
      path_offset=path_offset,
    )
    self._last_command = command
    return command
