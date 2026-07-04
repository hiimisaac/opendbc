from __future__ import annotations

from dataclasses import dataclass
import math


FORD_PATH_C0_CAN_CLIP = (-4.61, 4.60)
FORD_PATH_C1_CAN_CLIP = (-0.475, 0.497)

FORD_MODEL_DLOOK_TIME = 1.0
FORD_MODEL_DLOOK_MIN = 7.0
FORD_MODEL_C0_HIGH_SPEED_LOOKAHEAD = 6.0
FORD_MODEL_C0_SPEED_BP = (11.0, 14.0)

FORD_PATH_C0_RESIDUAL_SPEED_BP = (0.0, 5.0, 15.0, 30.0)
FORD_PATH_C0_RESIDUAL_GAIN = (0.55, 0.55, 0.35, 0.20)
FORD_PATH_C1_RATE_BP = (5.0, 25.0)
FORD_PATH_C1_RATE = (0.50, 0.50)

FORD_PATH_CURVATURE_THRESHOLD = 0.0040
FORD_PATH_CURVATURE_RATE_THRESHOLD = 0.020

FORD_C2_MEMORY_LIMIT = 0.02
FORD_C2_MEMORY_STACK_GAIN = 0.35
FORD_C2_MEMORY_DECAY_TAU = 3.0
FORD_C2_DT = 0.05  # LateralMotionControl2 runs at 20Hz.
FORD_C2_MEMORY_OVERSHOOT = 0.0007


@dataclass(frozen=True)
class C2MemoryStep:
  command: float
  memory: float


def _clip(value: float, lo: float, hi: float) -> float:
  return lo if value < lo else (hi if value > hi else value)


def _interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
  if x <= xs[0]:
    return ys[0]
  if x >= xs[-1]:
    return ys[-1]

  for i in range(1, len(xs)):
    if x < xs[i]:
      t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
  return ys[-1]


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def _same_sign(a: float, b: float) -> bool:
  return a * b > 0.0


def _valid_model_path(model) -> bool:
  if model is None:
    return False
  try:
    return len(model.position.x) > 1 and len(model.position.x) == len(model.position.y) == len(model.orientation.z)
  except (AttributeError, TypeError):
    return False


def _offset_from_path_angle(path_angle: float, d_c0: float, d_look: float) -> float:
  return 0.5 * path_angle * d_c0 * d_c0 / max(d_look, 0.1)


def should_use_path_fallback(desired_curvature: float, desired_curvature_rate: float,
                             c2_memory: float, lat_active: bool) -> bool:
  if not lat_active:
    return False

  desired_curvature = _finite(desired_curvature)
  desired_curvature_rate = _finite(desired_curvature_rate)
  c2_memory = _finite(c2_memory)

  large_curvature = abs(desired_curvature) >= FORD_PATH_CURVATURE_THRESHOLD
  rapid_curvature_change = abs(desired_curvature_rate) >= FORD_PATH_CURVATURE_RATE_THRESHOLD
  c2_overshot_target = abs(c2_memory) > abs(desired_curvature) + FORD_C2_MEMORY_OVERSHOOT
  c2_wrong_direction = _same_sign(c2_memory, -desired_curvature) and abs(c2_memory) > FORD_C2_MEMORY_OVERSHOOT

  return large_curvature or rapid_curvature_change or c2_overshot_target or c2_wrong_direction


def c2_memory_decay_step(c2_memory: float, lat_active: bool) -> C2MemoryStep:
  if not lat_active:
    return C2MemoryStep(0.0, 0.0)

  c2_memory = _finite(c2_memory)
  decay = math.exp(-FORD_C2_DT / FORD_C2_MEMORY_DECAY_TAU)
  return C2MemoryStep(0.0, _clip(c2_memory * decay, -FORD_C2_MEMORY_LIMIT, FORD_C2_MEMORY_LIMIT))


def c2_memory_step(commanded_curvature: float, c2_memory: float, lat_active: bool) -> C2MemoryStep:
  if not lat_active:
    return C2MemoryStep(0.0, 0.0)

  commanded_curvature = _finite(commanded_curvature)
  decayed_memory = c2_memory_decay_step(c2_memory, lat_active).memory
  memory = _clip(decayed_memory + commanded_curvature * FORD_C2_MEMORY_STACK_GAIN,
                 -FORD_C2_MEMORY_LIMIT, FORD_C2_MEMORY_LIMIT)
  return C2MemoryStep(commanded_curvature, memory)


def path_from_curvature(desired_curvature: float, v_ego: float, path_angle_last: float,
                        lat_active: bool, c2_memory: float = 0.0) -> tuple[float, float]:
  if not lat_active:
    return 0.0, 0.0

  desired_curvature = _finite(desired_curvature)
  c2_memory = _finite(c2_memory)
  v_ego = _finite(v_ego)

  d_look = max(v_ego * FORD_MODEL_DLOOK_TIME, FORD_MODEL_DLOOK_MIN)
  d_c0 = _interp(v_ego, FORD_MODEL_C0_SPEED_BP, (d_look, FORD_MODEL_C0_HIGH_SPEED_LOOKAHEAD))

  residual_curvature = desired_curvature - c2_memory
  path_angle = residual_curvature * d_look
  path_offset = 0.5 * residual_curvature * d_c0 * d_c0

  c1_rate = _interp(v_ego, FORD_PATH_C1_RATE_BP, FORD_PATH_C1_RATE)
  path_angle = _clip(path_angle, path_angle_last - c1_rate, path_angle_last + c1_rate)

  return (
    _clip(path_offset, *FORD_PATH_C0_CAN_CLIP),
    _clip(path_angle, *FORD_PATH_C1_CAN_CLIP),
  )


def path_from_model(model, desired_curvature: float, v_ego: float, path_angle_last: float,
                    lat_active: bool, c2_memory: float = 0.0) -> tuple[float, float]:
  if not _valid_model_path(model):
    return path_from_curvature(desired_curvature, v_ego, path_angle_last, lat_active, c2_memory)
  if not lat_active:
    return 0.0, 0.0

  desired_curvature = _finite(desired_curvature)
  c2_memory = _finite(c2_memory)
  v_ego = _finite(v_ego)

  d_look = max(v_ego * FORD_MODEL_DLOOK_TIME, FORD_MODEL_DLOOK_MIN)
  d_c0 = _interp(v_ego, FORD_MODEL_C0_SPEED_BP, (d_look, FORD_MODEL_C0_HIGH_SPEED_LOOKAHEAD))

  x_pts = model.position.x
  model_path_angle = _interp(d_look, x_pts, model.orientation.z) - c2_memory * d_look
  model_path_offset = _interp(d_c0, x_pts, model.position.y) - 0.5 * c2_memory * d_c0 * d_c0

  path_offset = _offset_from_path_angle(model_path_angle, d_c0, d_look)
  residual_gain = _interp(v_ego, FORD_PATH_C0_RESIDUAL_SPEED_BP, FORD_PATH_C0_RESIDUAL_GAIN)
  path_offset += (model_path_offset - path_offset) * residual_gain

  c1_rate = _interp(v_ego, FORD_PATH_C1_RATE_BP, FORD_PATH_C1_RATE)
  path_angle = _clip(model_path_angle, path_angle_last - c1_rate, path_angle_last + c1_rate)

  return (
    _clip(path_offset, *FORD_PATH_C0_CAN_CLIP),
    _clip(path_angle, *FORD_PATH_C1_CAN_CLIP),
  )
