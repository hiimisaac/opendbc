from __future__ import annotations

from dataclasses import dataclass
import math


# Lightweight Ford CAN-FD path controller, matching the recent sp-dev-c3 setup.
# It emits c0/c1 residuals around a memory-estimated c2 carrier. The controller
# sends these with inverted CAN sign convention.
FORD_PATH_C0_CAN_CLIP = (-4.61, 4.60)
FORD_PATH_C1_CAN_CLIP = (-0.475, 0.497)
FORD_PATH_C2_CAN_CLIP = (-0.02, 0.02)

FORD_MODEL_DLOOK_TIME = 1.0
FORD_MODEL_DLOOK_MIN = 7.0
FORD_MODEL_C0_HIGH_SPEED_LOOKAHEAD = 6.0
FORD_MODEL_C0_SPEED_BP = (11.0, 14.0)

FORD_PATH_FEEDBACK_SPEED_BP = (0.0, 5.0, 15.0, 30.0)
FORD_PATH_C1_FEEDBACK = (0.0, 2.0, 5.0, 6.0)
FORD_PATH_CURVATURE_ERROR = 0.006

FORD_PATH_C0_RESIDUAL_SPEED_BP = (0.0, 5.0, 15.0, 30.0)
FORD_PATH_C0_RESIDUAL_GAIN = (0.55, 0.55, 0.35, 0.20)

FORD_PATH_C1_RATE_BP = (5.0, 25.0)
FORD_PATH_C1_RATE = (0.50, 0.50)

FORD_C2_MEMORY_LIMIT = 0.02
FORD_C2_MEMORY_STACK_GAIN = 0.35
FORD_C2_MEMORY_DECAY_TAU = 3.0
FORD_C2_HOLD_MARGIN = 0.0005
FORD_C2_TARGET_DEADBAND = 0.0002
FORD_C2_DT = 0.05  # LateralMotionControl2 runs at 20Hz.


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


def _valid_model_path(model) -> bool:
  if model is None:
    return False
  try:
    return len(model.position.x) > 1 and len(model.position.x) == len(model.position.y) == len(model.orientation.z)
  except (AttributeError, TypeError):
    return False


def _offset_from_path_angle(path_angle: float, d_c0: float, d_look: float) -> float:
  return 0.5 * path_angle * d_c0 * d_c0 / max(d_look, 0.1)


def _same_sign(a: float, b: float) -> bool:
  return a * b > 0.0


def c2_memory_step(desired_curvature: float, current_curvature: float, c2_memory: float,
                   lat_active: bool) -> C2MemoryStep:
  """Return the next c2 pump command and estimated stored c2 curvature.

  Ford c2 behaves like a memory-bearing channel on this setup: the DBC range is
  the per-frame command, not the total stored steering authority. This controller
  charges c2 toward the requested curvature while it is useful, but avoids
  same-direction pumping when the request is unwinding or the car is already
  holding more curvature than requested.
  """
  if not lat_active:
    return C2MemoryStep(0.0, 0.0)

  desired_curvature = _finite(desired_curvature)
  current_curvature = _finite(current_curvature)
  c2_memory = _finite(c2_memory)

  decay = math.exp(-FORD_C2_DT / FORD_C2_MEMORY_DECAY_TAU)
  decayed_memory = c2_memory * decay
  target = _clip(desired_curvature, -FORD_C2_MEMORY_LIMIT, FORD_C2_MEMORY_LIMIT)

  holding_past_target = abs(current_curvature) > abs(target) + FORD_C2_HOLD_MARGIN and \
                        (_same_sign(current_curvature, target) or abs(target) < FORD_C2_TARGET_DEADBAND)
  memory_past_target = _same_sign(decayed_memory, target) and abs(decayed_memory) >= abs(target)
  stale_opposite_memory = _same_sign(decayed_memory, -target) and abs(decayed_memory) > FORD_C2_TARGET_DEADBAND

  if abs(target) < FORD_C2_TARGET_DEADBAND or holding_past_target or memory_past_target or stale_opposite_memory:
    command = 0.0
  else:
    command = _clip(target - decayed_memory, *FORD_PATH_C2_CAN_CLIP)

  memory = _clip(decayed_memory + command * FORD_C2_MEMORY_STACK_GAIN,
                 -FORD_C2_MEMORY_LIMIT, FORD_C2_MEMORY_LIMIT)
  return C2MemoryStep(command, memory)


def lightweight_path_from_curvature(desired_curvature: float, v_ego: float,
                                    path_angle_last: float, lat_active: bool,
                                    c2_memory: float = 0.0) -> tuple[float, float]:
  """Return Ford path offset and path angle for the lightweight path controller.

  This fallback synthesizes a constant-curvature model path when live model
  samples are unavailable. c0/c1 are residuals after removing the estimated c2
  memory arc, so c2 and c0/c1 do not express the same curve twice.
  """
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


def lightweight_path_from_model(model, desired_curvature: float, current_curvature: float, v_ego: float,
                                path_angle_last: float, lat_active: bool,
                                c2_memory: float = 0.0) -> tuple[float, float]:
  """Return Ford residual c0/c1 from the model path plus bounded c1 feedback.

  The model supplies the path shape Ford's PSCM wants: c1 from orientation.z at
  a far lookahead and a damped c0 companion from position.y at a shorter
  speed-dependent lookahead. The estimated c2 memory arc is removed from both
  terms before c0/c1 are emitted. Curvature error assists c1 only so c0 doesn't
  become a persistent artificial lateral offset.
  """
  if not _valid_model_path(model):
    return lightweight_path_from_curvature(desired_curvature, v_ego, path_angle_last, lat_active, c2_memory)
  if not lat_active:
    return 0.0, 0.0

  desired_curvature = _finite(desired_curvature)
  current_curvature = _finite(current_curvature)
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

  curvature_error = _clip(desired_curvature - current_curvature, -FORD_PATH_CURVATURE_ERROR, FORD_PATH_CURVATURE_ERROR)
  path_angle = model_path_angle + curvature_error * _interp(v_ego, FORD_PATH_FEEDBACK_SPEED_BP, FORD_PATH_C1_FEEDBACK)

  c1_rate = _interp(v_ego, FORD_PATH_C1_RATE_BP, FORD_PATH_C1_RATE)
  path_angle = _clip(path_angle, path_angle_last - c1_rate, path_angle_last + c1_rate)

  return (
    _clip(path_offset, *FORD_PATH_C0_CAN_CLIP),
    _clip(path_angle, *FORD_PATH_C1_CAN_CLIP),
  )
