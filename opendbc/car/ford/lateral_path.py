"""Ford steering feedback helpers."""

from collections import deque
import math


FORD_PATH_DT = 0.05
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.35


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def driver_steering_opposes_command(steering_pressed: bool, steering_torque: float,
                                     steering_angle_error_deg: float) -> bool:
  """Select cooperative path tracking when the driver opposes the request."""
  if not steering_pressed:
    return False

  steering_torque = _finite(steering_torque)
  steering_angle_error_deg = _finite(steering_angle_error_deg)
  return steering_angle_error_deg == 0.0 or steering_torque * steering_angle_error_deg < 0.0


class SteeringAngleProjector:
  """Project steering angle from a short, fixed-rate measurement window."""

  def __init__(self, sample_dt: float = FORD_PATH_DT,
               horizon: float = FORD_PATH_ANGLE_PROJECTION_HORIZON):
    self.sample_dt = max(_finite(sample_dt), FORD_PATH_DT)
    self.horizon = max(_finite(horizon), 0.0)
    self.samples: deque[float] = deque(
      maxlen=max(round(self.horizon / self.sample_dt) + 1, 2),
    )

  def update(self, actual_angle_deg: float) -> float:
    actual_angle_deg = _finite(actual_angle_deg, self.samples[-1] if self.samples else 0.0)
    self.samples.append(actual_angle_deg)
    sample_time = (len(self.samples) - 1) * self.sample_dt
    if sample_time <= 0.0:
      return actual_angle_deg

    steering_rate_deg_s = (self.samples[-1] - self.samples[0]) / sample_time
    return actual_angle_deg + steering_rate_deg_s * self.horizon
