import unittest

from opendbc.car.ford.lateral_path import (
  SteeringAngleProjector,
  driver_steering_opposes_command,
)


class TestLateralPath(unittest.TestCase):
  def test_driver_override_only_when_torque_opposes_angle_error(self):
    assert driver_steering_opposes_command(True, 1.125, -15.0)
    assert not driver_steering_opposes_command(True, 1.125, 15.0)
    assert not driver_steering_opposes_command(False, 1.125, -15.0)
    assert driver_steering_opposes_command(True, 1.125, 0.0)

  def test_angle_projector_extrapolates_recent_wheel_motion(self):
    projector = SteeringAngleProjector()
    projected = 0.0
    for angle in range(8):
      projected = projector.update(float(angle))

    assert projected > 7.0

  def test_angle_projector_holds_a_steady_wheel(self):
    projector = SteeringAngleProjector()
    for _ in range(10):
      projected = projector.update(12.0)

    assert projected == 12.0
