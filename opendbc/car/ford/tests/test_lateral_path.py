from opendbc.car.ford.lateral_path import (
  SteeringAngleProjector,
  driver_steering_opposes_command,
)


def test_driver_override_requires_opposing_steering_torque():
  assert driver_steering_opposes_command(True, 1.125, -15.0)
  assert not driver_steering_opposes_command(True, 1.125, 15.0)
  assert not driver_steering_opposes_command(False, 1.125, -15.0)


def test_steering_angle_projector_extrapolates_recent_wheel_motion():
  projector = SteeringAngleProjector()
  projected = 0.0
  for angle in (0.0, 1.0, 2.0, 3.0):
    projected = projector.update(angle)

  assert projected > 3.0
