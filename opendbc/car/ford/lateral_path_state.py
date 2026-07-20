from collections import deque
import math


FORD_PATH_DT = 0.05  # LateralMotionControl2 runs at 20Hz
FORD_PATH_ANGLE_PROJECTION_HORIZON = 0.35  # s, measured wheel-motion lookahead
FORD_PATH_OVERRIDE_PROJECTION_HORIZON = 0.1  # s, responsive driver-intent detection
FORD_PATH_DRIVER_OVERRIDE_IMMEDIATE_TORQUE = 2.0  # Nm, 2x normal steering-pressed threshold
FORD_PATH_DRIVER_OVERRIDE_CONFIRM_FRAMES = 2  # 0.1 s at 20Hz, including the first weak sample


def _finite(value: float, fallback: float = 0.0) -> float:
  return float(value) if math.isfinite(value) else fallback


def projected_tracking_error(target: float, current: float, projected: float) -> float:
  """Discount tracking error only when measured motion is closing the target."""
  error = _finite(target) - _finite(current)
  projected_error = _finite(target) - _finite(projected)
  projected_motion = _finite(projected) - _finite(current)
  if error == 0.0 or projected_motion * error <= 0.0:
    return error
  if projected_error * error <= 0.0:
    return 0.0
  return projected_error if abs(projected_error) < abs(error) else error


def driver_steering_opposes_command(steering_pressed: bool, steering_torque: float,
                                     steering_angle_error_deg: float) -> bool:
  """Select cooperative path tracking when the driver opposes the request.

  Compare two signals in steering-wheel coordinates. Ford curvature has the
  opposite sign from steering angle, which made the old curvature comparison
  classify a driver helping the requested wheel motion as an override. With no
  requested wheel motion, any pressed input takes priority.
  """
  if not steering_pressed:
    return False

  steering_torque = _finite(steering_torque)
  steering_angle_error_deg = _finite(steering_angle_error_deg)
  if steering_angle_error_deg == 0.0:
    return True
  return steering_torque * steering_angle_error_deg < 0.0


class SteeringAngleProjector:
  """Project steering angle from a short, fixed-rate measurement window."""

  def __init__(self, sample_dt: float = FORD_PATH_DT,
               horizon: float = FORD_PATH_ANGLE_PROJECTION_HORIZON):
    self.sample_dt = max(_finite(sample_dt), FORD_PATH_DT)
    self.horizon = max(_finite(horizon), 0.0)
    sample_count = max(round(self.horizon / self.sample_dt) + 1, 2)
    self.samples: deque[float] = deque(maxlen=sample_count)

  def update(self, actual_angle_deg: float) -> float:
    fallback = self.samples[-1] if self.samples else 0.0
    actual_angle_deg = _finite(actual_angle_deg, fallback)
    self.samples.append(actual_angle_deg)
    sample_time = (len(self.samples) - 1) * self.sample_dt
    if sample_time <= 0.0:
      return actual_angle_deg

    steering_rate_deg_s = (self.samples[-1] - self.samples[0]) / sample_time
    return actual_angle_deg + steering_rate_deg_s * self.horizon


class DriverOverrideFilter:
  """Reject isolated weak opposing-torque pulses while the wheel closes target.

  Strong input and wheel motion away from the desired angle remain immediate.
  A real weak opposing input is delayed by only one LMC2 frame, then remains
  confirmed until the opposing signal releases.
  """

  def __init__(self, immediate_torque: float = FORD_PATH_DRIVER_OVERRIDE_IMMEDIATE_TORQUE,
               confirm_frames: int = FORD_PATH_DRIVER_OVERRIDE_CONFIRM_FRAMES):
    self.immediate_torque = max(_finite(immediate_torque), 0.0)
    self.confirm_frames = max(int(confirm_frames), 1)
    self.weak_opposition_frames = 0
    self.confirmed = False
    self.pending = False

  def reset(self) -> None:
    self.weak_opposition_frames = 0
    self.confirmed = False
    self.pending = False

  def update(self, steering_pressed: bool, steering_torque: float,
             steering_angle_error_deg: float, actual_angle_deg: float,
             projected_angle_deg: float) -> bool:
    steering_torque = _finite(steering_torque)
    steering_angle_error_deg = _finite(steering_angle_error_deg)
    actual_angle_deg = _finite(actual_angle_deg)
    projected_angle_deg = _finite(projected_angle_deg, actual_angle_deg)

    opposing = driver_steering_opposes_command(steering_pressed, steering_torque,
                                                steering_angle_error_deg)
    if not opposing:
      self.reset()
      return False

    projected_motion_deg = projected_angle_deg - actual_angle_deg
    projected_error_deg = steering_angle_error_deg - projected_motion_deg
    wheel_closing_target = projected_motion_deg * steering_angle_error_deg > 0.0 and \
                           abs(projected_error_deg) < abs(steering_angle_error_deg)

    if abs(steering_torque) >= self.immediate_torque or not wheel_closing_target:
      self.confirmed = True
      self.pending = False
      return True

    if self.confirmed:
      self.pending = False
      return True

    self.weak_opposition_frames += 1
    self.confirmed = self.weak_opposition_frames >= self.confirm_frames
    self.pending = not self.confirmed
    return self.confirmed
