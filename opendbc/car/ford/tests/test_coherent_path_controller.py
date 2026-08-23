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


def test_ordinary_c2_remains_unmodified_with_production_angle_inputs():
  command = ProjectedLatControlPath().update(
    path(c2=0.004), measured_curvature=0.003, v_ego=12.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0035,
    desired_angle_curvature=0.004,
  )

  assert command.coefficients() == (0.0, 0.0, 0.004, 0.0)


def test_large_maneuver_transfers_c2_fully_into_fast_coefficients():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert command.curvature == 0.0
  assert command.path_offset >= 0.0
  assert command.path_angle > 0.0
  assert math.isclose(command_equivalent_curvature(command), 0.014)


def test_wheel_ahead_does_not_reverse_a_model_supported_turn():
  controller = ProjectedLatControlPath()

  requested = path(c2=-0.0365018, c3=0.0084159)
  command = controller.update(
    requested, measured_curvature=-0.06, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.06,
    desired_angle_curvature=-0.03,
  )

  requested_curvature = 2.0 * requested.pathOffset / 7.0 ** 2 + 2.0 * requested.pathAngle / 7.0 + \
                        requested.curvature + requested.curvatureRate * 7.0 / 3.0
  assert requested_curvature < 0.0
  assert math.isclose(command_equivalent_curvature(command), requested_curvature, abs_tol=1e-9)


def test_fast_coefficients_fit_the_model_across_multiple_horizons():
  controller = ProjectedLatControlPath()

  requested = path(c2=-0.0625301, c3=0.0044267)
  command = controller.update(
    requested, measured_curvature=-0.08, v_ego=2.97,
    active=True, driver_override=False,
    projected_measured_curvature=-0.08,
  )

  horizons = (3.0, 5.0, 7.0, 10.0)
  errors = [
    command_equivalent_curvature(command, distance) - path_equivalent_curvature(requested, distance)
    for distance in horizons
  ]
  rms_error = math.sqrt(sum(error ** 2 for error in errors) / len(errors))

  assert math.isclose(command_equivalent_curvature(command), path_equivalent_curvature(requested), abs_tol=1e-9)
  assert rms_error < 0.05


def test_clipped_c3_cannot_overrun_the_active_preview_target():
  controller = ProjectedLatControlPath()

  requested = path(c2=-0.01, c3=0.005)
  command = controller.update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert path_equivalent_curvature(requested) > 0.0
  assert math.isclose(
    command_equivalent_curvature(command),
    path_equivalent_curvature(requested),
    abs_tol=1e-9,
  )


def test_large_maneuver_adds_a_coherent_fast_error_arc_while_behind():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.01,
  )

  assert math.isclose(command_equivalent_curvature(command), 0.044, abs_tol=1e-9)


def test_fast_error_arc_disappears_at_projected_arrival():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.01,
    desired_angle_curvature=0.01,
  )

  assert math.isclose(command_equivalent_curvature(command), 0.014, abs_tol=1e-9)


def test_fast_error_arc_scales_uniformly_to_coefficient_headroom():
  requested = path(c0=2.0, c1=0.4, c2=0.014)
  arrived = ProjectedLatControlPath().update(
    requested, measured_curvature=0.02, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.02,
    desired_angle_curvature=0.02,
  )
  behind = ProjectedLatControlPath().update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.02,
  )

  c0_delta = behind.path_offset - arrived.path_offset
  c1_delta = behind.path_angle - arrived.path_angle
  assert math.isclose(behind.path_angle, 0.497)
  assert c0_delta > 0.0
  assert math.isclose(c0_delta / c1_delta, 0.5 * 7.0)


def test_negative_fast_error_arc_scales_uniformly_to_asymmetric_headroom():
  requested = path(c0=-2.0, c1=-0.4, c2=-0.014)
  arrived = ProjectedLatControlPath().update(
    requested, measured_curvature=-0.02, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.02,
    desired_angle_curvature=-0.02,
  )
  behind = ProjectedLatControlPath().update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=-0.02,
  )

  c0_delta = behind.path_offset - arrived.path_offset
  c1_delta = behind.path_angle - arrived.path_angle
  assert math.isclose(behind.path_angle, -0.475)
  assert c0_delta < 0.0
  assert math.isclose(c0_delta / c1_delta, 0.5 * 7.0)


def test_large_desired_angle_gets_full_error_arc_with_modest_model_curvature():
  command = ProjectedLatControlPath().update(
    path(c2=0.004), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.014,
  )

  assert math.isclose(command.curvature, 0.004)
  assert math.isclose(command_equivalent_curvature(command), 0.046, abs_tol=1e-9)


def test_verified_large_undertracking_retains_fast_ownership():
  controller = ProjectedLatControlPath()

  command = controller.update(
    path(c2=0.008), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert 0.0 < command.curvature < 0.004
  assert command.path_offset >= 0.0
  assert command.path_angle > 0.0
  assert math.isclose(command_equivalent_curvature(command), 0.008)


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
  unconstrained = ProjectedLatControlPath().update(
    path(c2=0.008), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.02,
  )
  constrained = controller.update(
    path(c2=0.008), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.02,
    lat_ctl_limit=1,
  )
  released = controller.update(
    path(c2=0.0), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    lat_ctl_limit=2,
  )

  assert command_equivalent_curvature(unconstrained) > command_equivalent_curvature(baseline)
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


def test_driver_release_immediately_restores_the_model_path():
  controller = ProjectedLatControlPath()
  requested = path(c2=0.0117)
  controller.update(
    requested, measured_curvature=0.018, v_ego=5.0,
    active=True, driver_override=True,
    projected_measured_curvature=0.018,
  )

  resumed = controller.update(
    requested, measured_curvature=0.018, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.018,
    desired_angle_curvature=0.006,
  )

  assert math.isclose(
    command_equivalent_curvature(resumed),
    path_equivalent_curvature(requested),
    abs_tol=1e-9,
  )


def test_model_turn_exit_releases_without_retained_angle_authority():
  controller = ProjectedLatControlPath()
  controller.update(
    path(c2=-0.03, c3=0.006), measured_curvature=-0.01, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.01,
    desired_angle_curvature=-0.02,
  )

  released = controller.update(
    path(), measured_curvature=-0.02, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.015,
    desired_angle_curvature=0.0,
  )

  assert released.coefficients() == (0.0, 0.0, 0.0, 0.0)


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


def command_equivalent_curvature(command, distance: float = 7.0) -> float:
  c0, c1, c2, c3 = command.coefficients()
  return 2.0 * c0 / distance ** 2 + 2.0 * c1 / distance + c2 + c3 * distance / 3.0


def path_equivalent_curvature(requested, distance: float = 7.0) -> float:
  return 2.0 * requested.pathOffset / distance ** 2 + 2.0 * requested.pathAngle / distance + \
         requested.curvature + requested.curvatureRate * distance / 3.0


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
