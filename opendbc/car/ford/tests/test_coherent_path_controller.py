import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_path import driver_steering_opposes_command, SteeringAngleProjector
from opendbc.car.ford.lateral_path_projector import ProjectedLatControlPath


def path(c0: float = 0.0, c1: float = 0.0,
         c2: float = 0.0, c3: float = 0.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=c0,
    pathAngle=c1,
    curvature=c2,
    curvatureRate=c3,
  )


def test_ordinary_driving_is_c2_only():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.004), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert math.isclose(command.curvature, 0.004)
  assert command.curvature_rate == 0.0


def test_preview_authority_disappears_when_wheel_is_on_the_path():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.004), measured_curvature=0.004, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.004,
  )

  assert math.isclose(command.path_offset, 0.0, abs_tol=1e-12)
  assert math.isclose(command.path_angle, 0.0, abs_tol=1e-12)
  assert math.isclose(command.curvature, 0.004)


def test_ordinary_c2_is_not_contaminated_by_wheel_tracking_error():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.004), measured_curvature=0.006, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.006,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert math.isclose(command.curvature, 0.004)


def test_large_maneuver_transfers_c2_fully_into_fast_coefficients():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert command.curvature == 0.0
  assert command.path_offset < 0.0
  assert command.path_angle > 0.0
  assert math.isclose(command_equivalent_curvature(command), 0.028)


def test_small_spatial_slope_does_not_disturb_ordinary_c2():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.004, c3=0.0004), measured_curvature=0.004, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.004,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert math.isclose(command.curvature, 0.004)
  assert command.curvature_rate == 0.0


def test_model_supported_reversal_uses_fast_coefficients_and_zeros_c2():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c0=0.12, c2=-0.002), measured_curvature=-0.006, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.006,
  )

  assert command.curvature == 0.0
  assert command_equivalent_curvature(command) > 0.0


def test_pscm_limit_blocks_outward_growth_but_allows_release():
  controller = ProjectedLatControlPath()
  baseline = controller.update(
    path(c2=0.004), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
  )
  constrained = controller.update(
    path(c2=0.008), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    lat_ctl_limit=1,
  )
  released = controller.update(
    path(c2=0.0), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    lat_ctl_limit=2,
  )

  assert abs(command_equivalent_curvature(constrained)) <= abs(command_equivalent_curvature(baseline)) + 1e-9
  assert command_equivalent_curvature(released) < command_equivalent_curvature(constrained)


def test_driver_override_and_release_are_bumpless():
  controller = ProjectedLatControlPath()
  override = controller.update(
    path(c2=0.012), measured_curvature=0.007, v_ego=5.0,
    active=True, driver_override=True,
    projected_measured_curvature=0.007,
  )
  resumed = controller.update(
    path(c2=0.012), measured_curvature=0.007, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.007,
  )

  assert override.path_offset > 0.0
  assert resumed.path_angle > 0.0
  assert override.curvature == 0.0
  assert resumed.curvature == 0.0
  assert command_equivalent_curvature(override) > 0.0
  assert command_equivalent_curvature(resumed) > 0.0


def test_left_and_right_paths_are_symmetric():
  left = ProjectedLatControlPath().update(
    path(c2=0.005, c3=0.0003), measured_curvature=0.001, v_ego=8.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.001,
  )
  right = ProjectedLatControlPath().update(
    path(c2=-0.005, c3=-0.0003), measured_curvature=-0.001, v_ego=8.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.001,
  )

  for left_value, right_value in zip(left.coefficients(), right.coefficients(), strict=True):
    assert math.isclose(left_value, -right_value)


def command_equivalent_curvature(command) -> float:
  c0, c1, c2, c3 = command.coefficients()
  return 2.0 * c0 / 7.0 ** 2 + 2.0 * c1 / 7.0 + c2 + c3 * 7.0 / 3.0


def test_driver_override_requires_torque_opposing_the_angle_request():
  assert driver_steering_opposes_command(True, 1.125, -15.0)
  assert not driver_steering_opposes_command(True, 1.125, 15.0)
  assert not driver_steering_opposes_command(False, 1.125, -15.0)


def test_angle_projector_uses_the_fixed_measurement_window():
  projector = SteeringAngleProjector()
  projected = 0.0
  for angle in range(8):
    projected = projector.update(float(angle))

  assert math.isclose(projected, 14.0)
