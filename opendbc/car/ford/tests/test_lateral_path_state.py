import math

from opendbc.car.ford.lateral_path_state import (
  DriverOverrideFilter,
  FORD_PATH_OVERRIDE_PROJECTION_HORIZON,
  SteeringAngleProjector,
  driver_steering_opposes_command,
  projected_tracking_error,
)


def test_projection_only_discounts_motion_closing_target():
  assert math.isclose(projected_tracking_error(0.02, 0.01, 0.015), 0.005)
  assert math.isclose(projected_tracking_error(0.02, 0.01, 0.005), 0.01)
  assert projected_tracking_error(0.02, 0.01, 0.025) == 0.0


def test_steering_angle_projector_uses_fixed_rate_window():
  projector = SteeringAngleProjector(sample_dt=0.05, horizon=0.35)
  projected = 0.0
  for angle in range(8):
    projected = projector.update(float(angle))

  assert math.isclose(projected, 14.0)


def test_driver_override_can_use_short_projection_horizon():
  projector = SteeringAngleProjector(horizon=FORD_PATH_OVERRIDE_PROJECTION_HORIZON)
  for angle in (0.0, 1.0, 2.0):
    projected = projector.update(angle)

  assert math.isclose(projected, 4.0)


def test_driver_override_direction_is_in_wheel_coordinates():
  assert not driver_steering_opposes_command(True, -1.125, -15.0)
  assert not driver_steering_opposes_command(True, 1.125, 15.0)
  assert driver_steering_opposes_command(True, 1.125, -15.0)
  assert driver_steering_opposes_command(True, -1.125, 15.0)
  assert driver_steering_opposes_command(True, 1.125, 0.0)
  assert not driver_steering_opposes_command(False, 1.125, -15.0)


def test_single_weak_road_bump_does_not_override_closing_wheel():
  override_filter = DriverOverrideFilter()

  assert not override_filter.update(True, -1.0625, 95.1, 116.7, 122.1)
  assert override_filter.pending
  assert not override_filter.update(False, -0.9375, 99.5, 120.1, 125.5)
  assert not override_filter.pending


def test_sustained_weak_opposition_overrides_on_second_frame():
  override_filter = DriverOverrideFilter()

  assert not override_filter.update(True, -1.125, 80.0, 120.0, 125.0)
  assert override_filter.update(True, -1.125, 75.0, 125.0, 130.0)
  assert override_filter.update(True, -1.125, 70.0, 130.0, 135.0)


def test_strong_or_diverging_opposition_overrides_immediately():
  assert DriverOverrideFilter().update(True, -2.0, 80.0, 120.0, 125.0)
  assert DriverOverrideFilter().update(True, -1.125, 80.0, 120.0, 115.0)
