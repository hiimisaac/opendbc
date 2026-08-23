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
  assert math.isclose(command.path_offset, 0.5 * 0.014 * 7.0 ** 2)
  assert math.isclose(command.path_angle, 0.014 * 7.0)


def test_large_action_is_encoded_as_a_complete_fast_arc_at_target():
  command = ProjectedLatControlPath().update(
    path(c2=0.06), measured_curvature=0.06, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.06,
    desired_angle_curvature=0.06,
  )

  assert math.isclose(command.path_offset, 0.5 * 0.06 * 7.0 ** 2)
  assert math.isclose(command.path_angle, 0.06 * 7.0)
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.0


def test_moderate_steady_turn_remains_on_c2_without_spatial_demand():
  command = ProjectedLatControlPath().update(
    path(c2=0.014), measured_curvature=0.014, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.014,
    desired_angle_curvature=0.014,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert math.isclose(command.curvature, 0.014)
  assert command.curvature_rate == 0.0


def test_complete_fast_arc_tapers_but_does_not_drop_when_wheel_is_beyond_target():
  requested = path(c2=0.06)
  at_target = ProjectedLatControlPath().update(
    requested, measured_curvature=0.05, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.05,
    desired_angle_curvature=0.05,
  )
  beyond_target = ProjectedLatControlPath().update(
    requested, measured_curvature=0.07, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.07,
    desired_angle_curvature=0.05,
  )

  assert 0.0 < beyond_target.path_offset < at_target.path_offset
  assert 0.0 < beyond_target.path_angle < at_target.path_angle
  assert beyond_target.curvature == 0.0


def test_extreme_overshoot_releases_the_fast_arc_completely():
  command = ProjectedLatControlPath().update(
    path(c2=0.06), measured_curvature=0.20, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.20,
    desired_angle_curvature=0.05,
  )

  assert command.coefficients() == (0.0, 0.0, 0.0, 0.0)


def test_inward_model_slope_bounds_arrived_arc_to_the_desired_corridor():
  command = ProjectedLatControlPath().update(
    path(c2=0.0786, c3=-0.011),
    measured_curvature=0.095, v_ego=4.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.080,
    desired_angle_curvature=0.079,
  )

  assert command.path_offset > 0.0
  assert command.path_angle > 0.0
  assert command_equivalent_curvature(command) <= 0.079 + 0.006 + 1e-9


def test_verified_desired_angle_shortfall_extends_both_fast_channels():
  command = ProjectedLatControlPath().update(
    path(c2=0.04, c3=0.009),
    measured_curvature=0.024, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.025,
    desired_angle_curvature=0.04,
  )

  assert command.path_offset > 1.3
  assert command.path_angle > 0.35
  assert command.curvature == 0.0


def test_coherent_model_geometry_sets_the_fast_arc_target():
  command = ProjectedLatControlPath().update(
    path(c0=0.2, c1=0.05, c2=0.002, c3=0.001),
    measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert command.path_offset > 0.12
  assert command.path_angle > 0.03
  assert command.curvature == 0.0


def test_c2_handoff_keeps_the_fast_arc_until_the_anchor_is_available():
  controller = ProjectedLatControlPath()
  requested = path(c2=0.014)
  behind = controller.update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.014,
  )
  arrived = controller.update(
    requested, measured_curvature=0.014, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.014,
    desired_angle_curvature=0.014,
  )

  assert arrived.curvature > 0.0
  assert command_equivalent_curvature(arrived) > 0.035
  assert arrived.path_offset > 0.45 * behind.path_offset
  assert arrived.path_angle > 0.45 * behind.path_angle


def test_pscm_limit_does_not_collapse_an_incomplete_c2_handoff():
  requested = path(c2=0.014)
  no_limit = ProjectedLatControlPath()
  limited = ProjectedLatControlPath()
  for controller in (no_limit, limited):
    controller.update(
      requested, measured_curvature=0.0, v_ego=5.0,
      active=True, driver_override=False,
      projected_measured_curvature=0.0,
      desired_angle_curvature=0.014,
    )

  expected = no_limit.update(
    requested, measured_curvature=0.013, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.013,
    desired_angle_curvature=0.014,
  )
  constrained = limited.update(
    requested, measured_curvature=0.013, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.013,
    desired_angle_curvature=0.014,
    lat_ctl_limit=2,
  )

  assert command_equivalent_curvature(constrained) > 0.035
  assert constrained.coefficients() == expected.coefficients()


def test_opposing_c3_waits_for_arrival_and_then_unwinds_gradually():
  controller = ProjectedLatControlPath()
  requested = path(c2=0.04, c3=-0.003)
  behind = controller.update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.04,
  )
  arrived = controller.update(
    requested, measured_curvature=0.04, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.04,
    desired_angle_curvature=0.04,
  )

  assert behind.curvature_rate == 0.0
  assert -0.00021 < arrived.curvature_rate < 0.0


def test_wheel_ahead_does_not_reverse_a_model_supported_turn():
  controller = ProjectedLatControlPath()

  requested = path(c2=-0.0365018, c3=0.0084159)
  command = controller.update(
    requested, measured_curvature=-0.06, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.06,
    desired_angle_curvature=-0.03,
  )

  assert command_equivalent_curvature(command) < 0.0
  assert abs(command_equivalent_curvature(command)) <= abs(requested.curvature) + 1e-3


def test_inward_c3_cannot_reverse_the_active_model_arc():
  controller = ProjectedLatControlPath()

  requested = path(c2=-0.01, c3=0.005)
  command = controller.update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert command_equivalent_curvature(command) < 0.0
  assert command.curvature_rate > 0.0


def test_arrival_removes_tracking_extension_without_zeroing_the_model_arc():
  requested = path(c2=0.014)
  behind = ProjectedLatControlPath().update(
    requested, measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.01,
  )
  arrived = ProjectedLatControlPath().update(
    requested, measured_curvature=0.01, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.01,
    desired_angle_curvature=0.01,
  )

  assert 0.0 < command_equivalent_curvature(arrived) < command_equivalent_curvature(behind)


def test_desired_angle_alone_cannot_invent_model_authority():
  command = ProjectedLatControlPath().update(
    path(c2=0.004), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.014,
  )

  assert command.coefficients() == (0.0, 0.0, 0.004, 0.0)


def test_verified_large_undertracking_transfers_partial_c2_to_fast_arc():
  command = ProjectedLatControlPath().update(
    path(c2=0.008), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
  )

  assert 0.0 < command.curvature < 0.004
  assert command.path_offset > 0.0
  assert command.path_angle > 0.0
  assert command_equivalent_curvature(command) > 0.008


def test_fast_arc_is_stronger_while_the_wheel_is_behind():
  behind = ProjectedLatControlPath().update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.01,
  )
  arrived = ProjectedLatControlPath().update(
    path(c2=0.014), measured_curvature=0.0, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.01,
    desired_angle_curvature=0.01,
  )

  assert command_equivalent_curvature(behind) > command_equivalent_curvature(arrived)


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


def test_model_supported_reversal_uses_fast_coefficients():
  command = ProjectedLatControlPath().update(
    path(c2=0.04, c3=0.006), measured_curvature=-0.006, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=-0.005,
    desired_angle_curvature=0.04,
  )

  assert command.curvature == 0.0
  assert command_equivalent_curvature(command) > 0.0


def test_pscm_limit_does_not_erase_the_bounded_model_path():
  controller = ProjectedLatControlPath()
  unconstrained = ProjectedLatControlPath().update(
    path(c2=0.04, c3=0.006), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.04,
  )
  constrained = controller.update(
    path(c2=0.04, c3=0.006), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.04,
    lat_ctl_limit=2,
  )
  released = controller.update(
    path(c2=0.0), measured_curvature=0.002, v_ego=5.0,
    active=True, driver_override=False,
    projected_measured_curvature=0.002,
    lat_ctl_limit=2,
  )

  assert constrained.coefficients() == unconstrained.coefficients()
  assert released.coefficients() == (0.0, 0.0, 0.0, 0.0)


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
