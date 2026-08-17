from types import SimpleNamespace

from opendbc.car.ford.lateral_path_projector import (
  PATH_C0_CONTINUATION_MARGIN,
  LateralPathCommand,
  ProjectedLatControlPath,
  _extend_c0_c1_for_geometry_shortfall,
  lmc2_control_utilization,
  _taper_stale_outward_preview,
)


def model(path_offset: float, path_angle: float, curvature: float = 0.0, curvature_rate: float = 0.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=path_offset,
    pathAngle=path_angle,
    curvature=curvature,
    curvatureRate=curvature_rate,
  )


def polynomial_model(curvature: float, curvature_rate: float = 0.0, lookahead: float = 7.0):
  return model(
    0.5 * curvature * 7.0 ** 2 + curvature_rate * 7.0 ** 3 / 6.0,
    curvature * lookahead + 0.5 * curvature_rate * lookahead ** 2,
    curvature,
    curvature_rate,
  )


def split_geometry_model(offset_curvature: float, angle_curvature: float,
                         curvature: float, curvature_rate: float = 0.0):
  return model(
    0.5 * offset_curvature * 7.0 ** 2,
    angle_curvature * 7.0,
    curvature,
    curvature_rate,
  )


def equivalent_curvature(command, distance: float) -> float:
  path_offset = getattr(command, "path_offset", getattr(command, "pathOffset", 0.0))
  path_angle = getattr(command, "path_angle", getattr(command, "pathAngle", 0.0))
  curvature_rate = getattr(command, "curvature_rate", getattr(command, "curvatureRate", 0.0))
  y = path_offset + path_angle * distance + 0.5 * command.curvature * distance ** 2 + \
      curvature_rate * distance ** 3 / 6.0
  return 2.0 * y / distance ** 2


def test_lmc2_control_utilization_tracks_strongest_coefficient_and_direction():
  half_positive_c2 = LateralPathCommand(True, 0.0, 0.0, 0.01, 0.0)
  half_negative_c2 = LateralPathCommand(True, 0.0, 0.0, -0.01, 0.0)
  full_positive_c3 = LateralPathCommand(True, 0.0, 0.0, 0.0, 0.001023)

  assert lmc2_control_utilization(half_positive_c2, 0) == 0.5
  assert lmc2_control_utilization(half_negative_c2, 0) == -0.5
  assert lmc2_control_utilization(full_positive_c3, 0) == 1.0


def test_lmc2_control_utilization_includes_pscm_limit_state():
  low_command = LateralPathCommand(True, 0.0, 0.0, 0.002, 0.0)
  zero_command = LateralPathCommand(True, 0.0, 0.0, 0.0, 0.0)

  assert lmc2_control_utilization(low_command, 0) == 0.1
  assert lmc2_control_utilization(low_command, 1) == 0.8
  assert lmc2_control_utilization(low_command, 2) == 1.0
  assert lmc2_control_utilization(low_command, 3) == 0.1
  assert lmc2_control_utilization(zero_command, 2) == 0.0


def test_feasible_steady_model_is_reproduced_by_c2():
  controller = ProjectedLatControlPath()
  delivered_curvature = 0.015
  target = model(
    0.5 * delivered_curvature * 7.0 ** 2,
    delivered_curvature * 7.0,
    delivered_curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      target, delivered_curvature, 7.0, True, False,
      desired_angle_curvature=delivered_curvature,
    )

  assert command is not None
  for distance in (3.0, 7.0, 15.0, 30.0):
    assert abs(equivalent_curvature(command, distance) - delivered_curvature) < 1e-6


def test_clipped_coefficients_remain_bounded_and_directionally_coherent():
  controller = ProjectedLatControlPath()
  target = model(-2.7, -0.60, -0.05, -0.002)

  command = None
  for _ in range(100):
    command = controller.update(target, -0.03, 7.0, True, False)

  assert command is not None
  assert -4.61 <= command.path_offset <= 4.60
  assert -0.475 <= command.path_angle <= 0.497
  assert -0.02 <= command.curvature <= 0.02
  assert -0.001024 <= command.curvature_rate <= 0.001023
  assert equivalent_curvature(command, 7.0) < 0.0


def test_large_turn_flushes_c2_and_projects_its_path_into_other_coefficients():
  controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.015)

  command = None
  for _ in range(100):
    command = controller.update(target, 0.012, 7.0, True, False)

  assert command is not None
  assert command.curvature == 0.0
  assert command.path_offset > target.pathOffset or command.path_angle > target.pathAngle


def test_changing_spatial_geometry_adds_c0_c1_without_replacing_c2():
  steady_controller = ProjectedLatControlPath()
  changing_controller = ProjectedLatControlPath()

  for _ in range(30):
    steady = steady_controller.update(
      polynomial_model(0.004), 0.004, 7.0, True, False,
      desired_angle_curvature=0.004,
    )
    changing = changing_controller.update(
      polynomial_model(0.004, 0.0005), 0.001, 7.0, True, False,
      projected_measured_curvature=0.001,
      desired_angle_curvature=0.004,
    )

  assert changing.curvature == steady.curvature
  assert abs(changing.path_offset) > abs(steady.path_offset)
  assert abs(changing.path_angle) > abs(steady.path_angle)


def test_steady_spatial_geometry_retains_c2_authority():
  controller = ProjectedLatControlPath()

  for _ in range(30):
    command = controller.update(
      polynomial_model(0.004, lookahead=15.0), 0.004, 15.0, True, False,
      desired_angle_curvature=0.004,
    )

  assert command.curvature > 0.002
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_coherent_upcoming_reversal_drains_conflicting_c2_early():
  controller = ProjectedLatControlPath()

  for _ in range(30):
    controller.update(
      polynomial_model(0.003), 0.003, 7.0, True, False,
      desired_angle_curvature=0.003,
    )

  reversal_path = polynomial_model(-0.004, -0.0003)
  reversal_path.curvature = 0.003
  reversal = controller.update(
    reversal_path, 0.003, 7.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=-0.003,
  )

  assert reversal.curvature > 0.0


def test_spatial_transfer_does_not_remove_opposite_direction_c2():
  transfer_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  steady = polynomial_model(0.003)
  reversal_path = polynomial_model(-0.004, -0.0003)
  reversal_path.curvature = 0.003

  for controller in (transfer_controller, reference_controller):
    for _ in range(30):
      controller.update(
        steady, 0.003, 7.0, True, False,
        desired_angle_curvature=0.003,
      )

  transfer = transfer_controller.update(
    reversal_path, 0.003, 7.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=-0.003,
    lat_ctl_limit=0,
  )
  reference = reference_controller.update(
    reversal_path, 0.003, 7.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=-0.003,
    lat_ctl_limit=3,
  )

  assert transfer.coefficients() == reference.coefficients()


def test_spatial_slope_does_not_drain_c2_during_asynchronous_relatch():
  controller = ProjectedLatControlPath()

  for _ in range(30):
    controller.update(
      polynomial_model(0.004), 0.004, 7.0, True, False,
      desired_angle_curvature=0.004,
    )

  relatch = controller.update(
    split_geometry_model(-0.008, 0.001, 0.004, -0.0005),
    0.004, 7.0, True, False,
    projected_measured_curvature=0.004,
    desired_angle_curvature=-0.004,
  )

  assert 0.002 < relatch.curvature <= 0.004


def test_small_spatial_noise_does_not_move_steady_authority_out_of_c2():
  steady_controller = ProjectedLatControlPath()
  noisy_controller = ProjectedLatControlPath()

  for _ in range(30):
    steady = steady_controller.update(
      polynomial_model(0.004, lookahead=15.0), 0.004, 15.0, True, False,
      desired_angle_curvature=0.004,
    )
    noisy = noisy_controller.update(
      polynomial_model(0.004, 0.00005, lookahead=15.0), 0.004, 15.0, True, False,
      desired_angle_curvature=0.004,
    )

  assert noisy.curvature == steady.curvature
  assert noisy.path_offset == steady.path_offset
  assert noisy.path_angle == steady.path_angle


def test_meaningful_model_exit_cannot_project_to_the_opposite_direction():
  controller = ProjectedLatControlPath()
  target = model(-0.0916, -0.0217, -0.0017, 0.0005)

  command = controller.update(
    target, -0.0046, 11.54, True, False,
    desired_angle_curvature=-0.0017,
  )

  assert equivalent_curvature(target, 7.0) < -0.005
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_projected_arrival_removes_only_c0_c1_correction():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  behind_controller = ProjectedLatControlPath()
  arrived_controller = ProjectedLatControlPath()

  behind_command = None
  arrived_command = None
  for _ in range(100):
    behind_command = behind_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )
    arrived_command = arrived_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.016,
      desired_angle_curvature=desired_curvature,
    )

  assert behind_command is not None
  assert arrived_command is not None
  assert behind_command.path_offset > arrived_command.path_offset
  assert behind_command.path_angle > arrived_command.path_angle
  assert abs(arrived_command.path_offset - target.pathOffset) < 1e-9
  assert abs(arrived_command.path_angle - target.pathAngle) < 1e-9


def test_spatial_preview_keeps_full_authority_while_wheel_is_behind():
  model_curvature = 0.07
  desired_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_curvature,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  assert command.path_offset > target.pathOffset
  assert command.path_angle >= target.pathAngle


def test_rising_spatial_turn_keeps_preview_beyond_lagging_action():
  controller = ProjectedLatControlPath()
  spatial_curvature = 0.04
  action_curvature = 0.002
  rising_turn = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    action_curvature,
    0.005,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      rising_turn, 0.01, 7.0, True, False,
      projected_measured_curvature=0.01,
      desired_angle_curvature=action_curvature,
    )

  assert command is not None
  assert command.path_offset == rising_turn.pathOffset
  assert command.path_angle == rising_turn.pathAngle
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.001023


def test_spatial_preview_is_symmetric_for_right_turns():
  # Keep the raw polynomial inside Ford's intentionally asymmetric signal
  # bounds so this isolates controller symmetry from DBC clipping.
  model_curvature = 0.04
  desired_curvature = 0.015
  left_target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_curvature,
    0.0005,
  )
  right_target = model(
    -left_target.pathOffset,
    -left_target.pathAngle,
    -left_target.curvature,
    -left_target.curvatureRate,
  )
  left_controller = ProjectedLatControlPath()
  right_controller = ProjectedLatControlPath()

  left = None
  right = None
  for _ in range(100):
    left = left_controller.update(
      left_target, desired_curvature, 7.0, True, False,
      projected_measured_curvature=desired_curvature,
      desired_angle_curvature=desired_curvature,
    )
    right = right_controller.update(
      right_target, -desired_curvature, 7.0, True, False,
      projected_measured_curvature=-desired_curvature,
      desired_angle_curvature=-desired_curvature,
    )

  assert left is not None
  assert right is not None
  assert right.valid == left.valid
  for left_coefficient, right_coefficient in zip(left.coefficients(), right.coefficients(), strict=True):
    assert abs(right_coefficient + left_coefficient) < 1e-12


def test_continuing_model_preview_extends_only_c0_after_current_angle_arrival():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.002,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert command.path_offset > target.pathOffset
  assert abs(command.path_angle - target.pathAngle) < 1e-9
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.001023

  bounded_preview_arrived = controller.update(
    target, 0.01, 7.0, True, False,
    projected_measured_curvature=desired_angle_curvature + 0.006,
    desired_angle_curvature=desired_angle_curvature,
  )

  assert abs(bounded_preview_arrived.path_offset - target.pathOffset) < 1e-9
  assert abs(bounded_preview_arrived.path_angle - target.pathAngle) < 1e-9


def test_continuing_model_preview_cannot_extend_c0_past_bounded_angle_corridor():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.002,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, desired_angle_curvature + 0.007, 7.0, True, False,
      projected_measured_curvature=desired_angle_curvature + 0.009,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert abs(command.path_offset - target.pathOffset) < 1e-9
  assert abs(command.path_angle - target.pathAngle) < 1e-9


def test_available_c3_does_not_spill_continuation_into_c0():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.0005,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert abs(command.path_offset - target.pathOffset) < 1e-9
  assert abs(command.path_angle - target.pathAngle) < 1e-9
  assert command.curvature_rate == target.curvatureRate


def test_pscm_envelope_reallocates_outward_c3_into_c0_while_wheel_is_behind():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  controllers = [ProjectedLatControlPath() for _ in range(3)]
  commands = [None, None, None]

  for _ in range(100):
    for limit_status, controller in enumerate(controllers):
      commands[limit_status] = controller.update(
        target, 0.005, 7.0, True, False,
        projected_measured_curvature=0.005,
        desired_angle_curvature=desired_curvature,
        lat_ctl_limit=limit_status,
      )

  clear, close, reached = commands
  assert clear is not None
  assert close is not None
  assert reached is not None
  assert clear.path_offset < close.path_offset < reached.path_offset
  assert clear.curvature_rate > close.curvature_rate > reached.curvature_rate
  assert abs(close.curvature_rate - 0.5 * clear.curvature_rate) < 1e-12
  assert abs(reached.curvature_rate) < 1e-12
  assert equivalent_curvature(clear, 3.0) < equivalent_curvature(close, 3.0) < equivalent_curvature(reached, 3.0)
  assert equivalent_curvature(clear, 7.0) < equivalent_curvature(close, 7.0) < equivalent_curvature(reached, 7.0)
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(close, 15.0)) < 1e-9
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(reached, 15.0)) < 1e-9


def test_pscm_envelope_does_not_reallocate_c3_after_projected_arrival():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  baseline_controller = ProjectedLatControlPath()
  controller = ProjectedLatControlPath()

  baseline = None
  reallocated = None
  for _ in range(100):
    baseline = baseline_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.016,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    reallocated = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=2,
    )

  arrived = controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.016,
    desired_angle_curvature=desired_curvature,
    lat_ctl_limit=2,
  )

  assert baseline is not None
  assert reallocated is not None
  assert reallocated != baseline
  assert arrived == baseline


def test_pscm_driver_limit_does_not_add_path_authority():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  clear_controller = ProjectedLatControlPath()
  driver_limit_controller = ProjectedLatControlPath()

  clear = None
  driver_limited = None
  for _ in range(100):
    clear = clear_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    driver_limited = driver_limit_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=3,
    )

  assert clear == driver_limited


def test_pscm_envelope_reallocation_is_symmetric_for_right_turns():
  desired_curvature = -0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    -0.003,
  )
  clear_controller = ProjectedLatControlPath()
  reached_controller = ProjectedLatControlPath()

  clear = None
  reached = None
  for _ in range(100):
    clear = clear_controller.update(
      target, -0.005, 7.0, True, False,
      projected_measured_curvature=-0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    reached = reached_controller.update(
      target, -0.005, 7.0, True, False,
      projected_measured_curvature=-0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=2,
    )

  assert clear is not None
  assert reached is not None
  assert abs(reached.path_offset) > abs(clear.path_offset)
  assert abs(reached.curvature_rate) < abs(clear.curvature_rate)
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(reached, 15.0)) < 1e-9


def test_desired_angle_shortfall_extends_c0_c1_beyond_model_geometry():
  model_curvature = 0.008
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    model_curvature,
    0.003,
  )
  model_controller = ProjectedLatControlPath()
  deeper_controller = ProjectedLatControlPath()

  model_command = None
  deeper_command = None
  for _ in range(100):
    model_command = model_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=model_curvature,
    )
    deeper_command = deeper_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=0.015,
    )

  assert model_command is not None
  assert deeper_command is not None
  assert deeper_command.path_offset > model_command.path_offset
  assert deeper_command.path_angle > model_command.path_angle


def test_projected_gate_preserves_model_correction_while_both_wheel_estimates_are_behind():
  model_curvature = 0.015
  desired_curvature = 0.010
  measured_curvature = 0.005
  tracking_correction = model_curvature - measured_curvature - 0.0005
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    model_curvature,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, measured_curvature, 7.0, True, False,
      projected_measured_curvature=measured_curvature,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  assert abs(command.path_offset - 0.5 * (model_curvature + tracking_correction) * 7.0 ** 2) < 1e-9
  assert abs(command.path_angle - (model_curvature + tracking_correction) * 7.0) < 1e-9


def test_projected_crossing_drops_correction_immediately_without_reversing_model():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  arrived = controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.016,
    desired_angle_curvature=desired_curvature,
  )

  assert arrived.path_offset < command.path_offset
  assert arrived.path_angle < command.path_angle
  assert abs(arrived.path_offset - target.pathOffset) < 1e-9
  assert abs(arrived.path_angle - target.pathAngle) < 1e-9


def test_projected_arrival_tapers_c0_c1_correction_without_a_command_step():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  just_outside_controller = ProjectedLatControlPath()
  just_inside_controller = ProjectedLatControlPath()

  just_outside = None
  just_inside = None
  for _ in range(100):
    just_outside = just_outside_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.01449,
      desired_angle_curvature=desired_curvature,
    )
    just_inside = just_inside_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.01451,
      desired_angle_curvature=desired_curvature,
    )

  assert just_outside is not None
  assert just_inside is not None
  assert equivalent_curvature(just_outside, 7.0) >= equivalent_curvature(just_inside, 7.0)
  assert equivalent_curvature(just_outside, 7.0) - equivalent_curvature(just_inside, 7.0) < 0.001


def test_measured_wheel_and_desired_angle_reject_stale_opposing_geometry():
  controller = ProjectedLatControlPath()
  stale_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, -0.0004)

  command = controller.update(
    stale_geometry, 0.01, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) <= 0.0


def test_c3_that_continues_turn_preserves_preview_through_action_conflict():
  controller = ProjectedLatControlPath()
  continuing_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, 0.003)

  command = controller.update(
    continuing_geometry, 0.01, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) > 0.0


def test_opposing_action_does_not_discard_model_preview_before_wheel_follows_it():
  controller = ProjectedLatControlPath()
  entering_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, 0.003)

  command = controller.update(
    entering_geometry, -0.003, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) > 0.0


def test_opposing_action_does_not_block_c0_c1_correction_toward_desired_angle():
  model_curvature = 0.015
  entering_geometry = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    -0.004,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      entering_geometry, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=model_curvature,
    )

  assert command is not None
  assert command.path_offset > entering_geometry.pathOffset
  assert command.path_angle > entering_geometry.pathAngle


def test_medium_curve_allocates_c2_once_without_opposing_preview_coefficients():
  controller = ProjectedLatControlPath()
  curvature = 0.005
  target = model(0.5 * curvature * 7.0 ** 2, curvature * 7.0, curvature)

  command = None
  for _ in range(100):
    command = controller.update(
      target, curvature, 7.0, True, False,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.curvature > 0.0
  assert command.path_offset >= 0.0
  assert command.path_angle >= 0.0


def test_overtracking_reduces_authority_without_zeroing_model_path():
  target = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, 0.015, 0.003)
  behind_controller = ProjectedLatControlPath()
  beyond_controller = ProjectedLatControlPath()

  behind = None
  beyond = None
  for _ in range(100):
    behind = behind_controller.update(
      target, 0.005, 7.0, True, False,
      desired_angle_curvature=0.015,
    )
    beyond = beyond_controller.update(
      target, 0.025, 7.0, True, False,
      desired_angle_curvature=0.015,
    )

  assert behind is not None
  assert beyond is not None
  assert 0.0 < equivalent_curvature(beyond, 7.0) < equivalent_curvature(behind, 7.0)


def test_delivered_gentle_path_uses_c2_without_duplicate_preview_terms():
  controller = ProjectedLatControlPath()
  gentle = model(0.5 * 0.003 * 7.0 ** 2, 0.003 * 15.0, 0.003)

  command = None
  for _ in range(100):
    command = controller.update(gentle, 0.003, 15.0, True, False)

  assert command is not None
  assert abs(command.path_offset) < 1e-6
  assert abs(command.path_angle) < 1e-6
  assert abs(command.curvature - 0.003) < 1e-6


def test_spatially_steady_high_curvature_stays_on_c2():
  controller = ProjectedLatControlPath()
  curvature = 0.012
  speed = 10.0
  steady_curve = model(
    0.5 * curvature * 7.0 ** 2,
    curvature * speed,
    curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      steady_curve, curvature, speed, True, False,
      projected_measured_curvature=curvature,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature == curvature
  assert command.curvature_rate == 0.0


def test_gentle_path_uses_only_reversible_c2_despite_noisy_preview():
  controller = ProjectedLatControlPath()
  gentle = model(
    0.5 * 0.0025 * 7.0 ** 2,
    0.0025 * 25.0,
    0.002,
    -0.0001,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      gentle, 0.002, 25.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.002,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature == gentle.curvature
  assert command.curvature_rate == 0.0


def test_gentle_c2_anchor_is_not_changed_by_measured_wheel_disturbance():
  controller = ProjectedLatControlPath()
  gentle = model(
    0.5 * 0.0025 * 7.0 ** 2,
    0.0025 * 25.0,
    0.002,
    -0.0001,
  )

  for _ in range(100):
    controller.update(
      gentle, 0.002, 25.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.002,
    )

  disturbed = controller.update(
    gentle, 0.0065, 25.0, True, False,
    projected_measured_curvature=-0.001,
    desired_angle_curvature=0.002,
  )

  assert disturbed.path_offset == 0.0
  assert disturbed.path_angle == 0.0
  assert disturbed.curvature == gentle.curvature
  assert disturbed.curvature_rate == 0.0


def test_spatial_slope_crossfades_once_from_c2_to_full_polynomial():
  commands = []
  curvature = 0.003
  for maneuver_demand in (0.003, 0.0045, 0.006):
    controller = ProjectedLatControlPath()
    target = model(
      0.5 * curvature * 7.0 ** 2,
      curvature * 7.0,
      curvature,
      maneuver_demand * 3.0 / 7.0,
    )
    command = None
    for _ in range(100):
      command = controller.update(
        target, curvature, 7.0, True, False,
        desired_angle_curvature=curvature,
      )
    commands.append(command)

  assert all(command is not None for command in commands)
  gentle, transition, full = commands
  assert abs(gentle.curvature - 0.003) < 1e-12
  assert abs(gentle.path_offset) < 1e-12
  assert abs(gentle.path_angle) < 1e-12
  assert abs(transition.curvature - 0.0015) < 1e-9
  assert transition.path_offset > 0.0
  assert transition.path_angle > 0.0
  assert full.curvature == 0.0
  assert full.path_offset > transition.path_offset
  assert full.path_angle > transition.path_angle


def test_c0_c1_carry_c2_handoff_until_slow_anchor_arrives():
  # Captured from a right-turn collapse after a brief driver correction. The
  # model's complete polynomial still asks for the turn, but maneuver demand
  # crosses below the C0/C1 threshold while C2 is restarting from zero.
  for direction in (1.0, -1.0):
    controller = ProjectedLatControlPath()
    strong = model(*(direction * value for value in (
      1.100723624, 0.283491492, 0.018789796, -0.000230324,
    )))
    controller.update(
      strong, direction * 0.023635519, 10.361, True, False,
      projected_measured_curvature=direction * 0.019882582,
      desired_angle_curvature=direction * 0.022392068,
    )

    override_samples = (
      (1.046459317, 0.268339455, 0.018728370, -0.000238370,
       0.020665589, 0.013989830, 0.022316933),
      (0.982645273, 0.257616222, 0.019059796, -0.000116112,
       0.020397064, 0.013695550, 0.022714867),
      (0.954809308, 0.256530553, 0.019226456, 0.000042498,
       0.020821513, 0.015184017, 0.022912957),
    )
    for c0, c1, c2, c3, measured, projected, desired in override_samples:
      controller.update(
        model(*(direction * value for value in (c0, c1, c2, c3))),
        direction * measured, 10.46, True, True,
        projected_measured_curvature=direction * projected,
        desired_angle_curvature=direction * desired,
      )

    release_samples = (
      (0.912020087, 0.246393055, 0.019684806, 0.000222714,
       0.021310644, 0.017298289, 0.023446815),
      (0.880617678, 0.238525912, 0.019948771, 0.000280214,
       0.021681554, 0.019768474, 0.023769662),
      (0.876937568, 0.241124481, 0.019665301, 0.000372190,
       0.021863119, 0.021730294, 0.023431917),
    )
    for c0, c1, c2, c3, measured, projected, desired in release_samples:
      command = controller.update(
        model(*(direction * value for value in (c0, c1, c2, c3))),
        direction * measured, 10.55, True, False,
        projected_measured_curvature=direction * projected,
        desired_angle_curvature=direction * desired,
      )
      # C0/C1 must bridge the missing C2 allocation rather than allowing the
      # total command to collapse almost to zero during the transfer.
      assert direction * equivalent_curvature(command, 7.0) >= 0.015


def test_coherent_reversal_shortfall_keeps_polynomial_authority_below_spatial_threshold():
  # Captured from a fast left-to-right reversal. The model path, desired
  # steering angle, and spatial slope all continue into the new turn while the
  # wheel is still well behind. C3's spatial demand falls just below the normal
  # 0.003 polynomial crossover between these samples.
  for direction in (1.0, -1.0):
    controller = ProjectedLatControlPath()
    building = model(*(direction * value for value in (0.027868, 0.025723, 0.005296, 0.000834)))
    continuing = model(*(direction * value for value in (0.051800, 0.034843, 0.006418, 0.000722)))

    building_command = controller.update(
      building, direction * -0.001164, 11.02, True, False,
      projected_measured_curvature=direction * -0.001164,
      desired_angle_curvature=direction * 0.006059,
    )
    continuing_command = controller.update(
      continuing, direction * 0.002249, 11.18, True, False,
      projected_measured_curvature=direction * 0.002249,
      desired_angle_curvature=direction * 0.007435,
    )

    assert direction * continuing_command.path_offset > 0.0
    assert direction * continuing_command.path_angle > 0.0
    assert abs(equivalent_curvature(continuing_command, 7.0)) >= \
           0.5 * abs(equivalent_curvature(building_command, 7.0))


def test_projected_overshoot_cannot_drop_coherent_reversal_before_measured_arrival():
  controller = ProjectedLatControlPath()
  continuing = model(0.051800, 0.034843, 0.006418, 0.000722)

  command = controller.update(
    continuing, 0.002249, 11.18, True, False,
    # A fast wheel transient can extrapolate beyond desired even while the
    # measured wheel remains substantially behind it.
    projected_measured_curvature=0.010000,
    desired_angle_curvature=0.007435,
  )

  assert command.path_offset > 0.0
  assert command.path_angle > 0.0


def test_measured_arrival_drops_reversal_shortfall_floor_immediately():
  controller = ProjectedLatControlPath()
  continuing = model(0.051800, 0.034843, 0.006418, 0.000722)

  command = controller.update(
    continuing, 0.007600, 11.18, True, False,
    projected_measured_curvature=0.002249,
    desired_angle_curvature=0.007435,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature > 0.0
  assert command.curvature_rate == 0.0


def test_coherent_c0_c1_shortfall_uses_polynomial_without_c3_support():
  # Captured while entering a left turn: C0/C1 and the desired steering angle
  # agree, both measured and projected steering remain behind, and the PSCM is
  # not close to its limit. C3 has already changed toward unwind, but must not
  # collapse the still-needed near-field geometry back to C2-only.
  controller = ProjectedLatControlPath()
  baseline_controller = ProjectedLatControlPath()
  target = model(-0.413027, -0.083532, -0.011260, 0.000259)

  command = controller.update(
    target, -0.010105, 9.23, True, False,
    projected_measured_curvature=-0.012174,
    desired_angle_curvature=-0.012917,
    lat_ctl_limit=0,
  )
  baseline = baseline_controller.update(
    target, -0.010105, 9.23, True, False,
    projected_measured_curvature=-0.012174,
    desired_angle_curvature=-0.012917,
    lat_ctl_limit=1,
  )

  assert command.path_offset < 0.0
  assert command.path_angle < 0.0
  # The polynomial now also substitutes for the slow C2 anchor during this
  # handoff. It may exceed the projected tracking-error correction alone, but
  # remains inside the desired-angle curvature that C2 is rebuilding toward.
  assert abs(equivalent_curvature(baseline, 7.0)) < abs(equivalent_curvature(command, 7.0)) <= 0.012917


def test_c0_c1_shortfall_extension_drops_at_projected_arrival():
  controller = ProjectedLatControlPath()
  target = model(-0.413027, -0.083532, -0.011260, 0.000259)

  command = controller.update(
    target, -0.010105, 9.23, True, False,
    projected_measured_curvature=-0.012917,
    desired_angle_curvature=-0.012917,
    lat_ctl_limit=0,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_c0_c1_shortfall_extension_defers_to_pscm_limit_status():
  controller = ProjectedLatControlPath()
  target = model(-0.413027, -0.083532, -0.011260, 0.0)

  command = controller.update(
    target, -0.010105, 9.23, True, False,
    projected_measured_curvature=-0.012174,
    desired_angle_curvature=-0.012917,
    lat_ctl_limit=1,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_c0_c1_shortfall_extension_stays_out_of_ordinary_edge_curve():
  # Captured from ordinary Edge steering below 10 degrees. C0/C1 are coherent,
  # but their geometry is not large enough to justify maneuver authority.
  controller = ProjectedLatControlPath()
  target = model(0.134285, 0.048518, 0.003200, 0.0)

  command = controller.update(
    target, 0.001500, 9.6, True, False,
    projected_measured_curvature=0.002335,
    desired_angle_curvature=0.003200,
    lat_ctl_limit=0,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_c0_c1_shortfall_extension_stays_out_of_ordinary_highway_curve():
  controller = ProjectedLatControlPath()
  target = model(0.1715, 0.14, 0.004, 0.0)

  command = controller.update(
    target, 0.0025, 20.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=0.004,
    lat_ctl_limit=0,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_c0_c1_shortfall_extension_uses_unused_model_geometry_at_full_residual_share():
  coefficients = (0.0, 0.0, 0.0, 0.0)
  raw_target = (0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, 0.015, 0.003)

  command = _extend_c0_c1_for_geometry_shortfall(
    coefficients,
    raw_target,
    desired_curvature=0.015,
    measured_curvature=0.005,
    projected_curvature=0.005,
    v_ego=7.0,
    valid=True,
    lat_ctl_limit=0,
    residual_share=1.0,
  )

  assert 0.0 < command[0] <= raw_target[0]
  assert 0.0 < command[1] <= raw_target[1]


def test_c0_c1_shortfall_extension_remains_zero_at_projected_arrival():
  coefficients = (0.0, 0.0, 0.0, 0.0)
  raw_target = (0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, 0.015, 0.003)

  command = _extend_c0_c1_for_geometry_shortfall(
    coefficients,
    raw_target,
    desired_curvature=0.015,
    measured_curvature=0.005,
    projected_curvature=0.015,
    v_ego=7.0,
    valid=True,
    lat_ctl_limit=0,
    residual_share=1.0,
  )

  assert command == coefficients


def test_opposite_side_reversal_uses_c0_until_the_wheel_changes_direction():
  # Captured from a low-speed S-turn. The action and spatial slope have entered
  # the new turn while C0/C1 still describe the preceding arc and the wheel is
  # physically delivering that old direction.
  target = model(-0.0549, -0.0063, 0.0033, 0.000515)
  reversal_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()

  reversal = reversal_controller.update(
    target, -0.00241, 6.6, True, False,
    projected_measured_curvature=-0.00170,
    desired_angle_curvature=0.00326,
    lat_ctl_limit=0,
  )
  reference = reference_controller.update(
    target, -0.00241, 6.6, True, False,
    projected_measured_curvature=-0.00170,
    desired_angle_curvature=0.00326,
    lat_ctl_limit=1,
  )

  assert reversal.path_offset > reference.path_offset
  assert reversal.path_angle == reference.path_angle
  assert equivalent_curvature(reversal, 7.0) > equivalent_curvature(reference, 7.0)


def test_opposite_side_reversal_c0_drops_when_projection_changes_direction():
  target = model(-0.0549, -0.0063, 0.0033, 0.000515)
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()

  crossed = controller.update(
    target, -0.00241, 6.6, True, False,
    projected_measured_curvature=0.0001,
    desired_angle_curvature=0.00326,
    lat_ctl_limit=0,
  )
  reference = reference_controller.update(
    target, -0.00241, 6.6, True, False,
    projected_measured_curvature=-0.00170,
    desired_angle_curvature=0.00326,
    lat_ctl_limit=1,
  )

  assert crossed.path_offset == reference.path_offset
  assert crossed.path_angle == reference.path_angle


def test_opposite_side_reversal_c0_ignores_small_action_sign_noise():
  target = model(-0.01, -0.002, 0.0015, 0.000515)
  controller = ProjectedLatControlPath()

  command = controller.update(
    target, -0.0015, 6.6, True, False,
    projected_measured_curvature=-0.001,
    desired_angle_curvature=0.0015,
    lat_ctl_limit=0,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_opposite_side_reversal_c0_is_symmetric():
  commands = []
  for direction in (1.0, -1.0):
    controller = ProjectedLatControlPath()
    target = model(*(direction * value for value in (-0.0549, -0.0063, 0.0033, 0.000515)))
    commands.append(controller.update(
      target, direction * -0.00241, 6.6, True, False,
      projected_measured_curvature=direction * -0.00170,
      desired_angle_curvature=direction * 0.00326,
      lat_ctl_limit=0,
    ))

  positive, negative = commands
  assert abs(positive.path_offset + negative.path_offset) < 1e-12
  assert abs(positive.path_angle + negative.path_angle) < 1e-12
  assert abs(equivalent_curvature(positive, 7.0) + equivalent_curvature(negative, 7.0)) < 1e-12


def test_opposite_side_reversal_c0_cannot_exceed_tracking_correction_budget():
  preceding_arc = model(-0.2, -0.04, 0.009, 0.0008)
  target = model(0.0525008, 0.0003996, -0.0087378, -0.0007653)
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  for candidate, limit in ((controller, 0), (reference_controller, 3)):
    candidate.update(
      preceding_arc, -0.003, 7.0, True, False,
      projected_measured_curvature=-0.003,
      desired_angle_curvature=0.009,
      lat_ctl_limit=limit,
    )

  command = controller.update(
    target, 0.00301, 4.68, True, False,
    projected_measured_curvature=0.00304,
    desired_angle_curvature=-0.009125,
    lat_ctl_limit=0,
  )
  reference = reference_controller.update(
    target, 0.00301, 4.68, True, False,
    projected_measured_curvature=0.00304,
    desired_angle_curvature=-0.009125,
    lat_ctl_limit=3,
  )

  added_curvature = equivalent_curvature(reference, 7.0) - equivalent_curvature(command, 7.0)
  assert 0.0 < added_curvature <= 0.006


def test_opposite_side_reversal_c0_support_is_not_speed_scheduled():
  target = model(0.0, 0.0, 0.0031, 0.00006)
  additions = []
  for speed in (7.0, 30.0):
    controller = ProjectedLatControlPath()
    reference_controller = ProjectedLatControlPath()
    command = controller.update(
      target, -0.001, speed, True, False,
      projected_measured_curvature=-0.0021,
      desired_angle_curvature=0.0031,
      lat_ctl_limit=0,
    )
    reference = reference_controller.update(
      target, -0.001, speed, True, False,
      projected_measured_curvature=-0.0021,
      desired_angle_curvature=0.0031,
      lat_ctl_limit=3,
    )
    additions.append(equivalent_curvature(command, 7.0) - equivalent_curvature(reference, 7.0))

  assert abs(additions[0] - additions[1]) < 1e-12


def test_opposite_side_reversal_c0_ignores_noisy_zero_measured_wheel():
  target = model(0.0, 0.0, 0.0031, 0.0003)
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()

  command = controller.update(
    target, -1e-12, 7.0, True, False,
    projected_measured_curvature=-0.0021,
    desired_angle_curvature=0.0031,
    lat_ctl_limit=0,
  )
  reference = reference_controller.update(
    target, -1e-12, 7.0, True, False,
    projected_measured_curvature=-0.0021,
    desired_angle_curvature=0.0031,
    lat_ctl_limit=3,
  )

  assert equivalent_curvature(command, 7.0) == equivalent_curvature(reference, 7.0)


def test_c0_c1_shortfall_extension_respects_aggregate_model_allocation():
  target = model(0.40618104697643687, 0.24009829608833588,
                 0.005067795021689749, 0.0006409813585709391)
  normal_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  args = (
    target, 0.009449273650111256, 28.965969542162057, True, False,
  )
  kwargs = {
    "projected_measured_curvature": -0.0062674601564500845,
    "desired_angle_curvature": 0.010909133211626072,
  }

  command = normal_controller.update(*args, lat_ctl_limit=0, **kwargs)
  extension_disabled = reference_controller.update(*args, lat_ctl_limit=3, **kwargs)

  assert command.path_offset == extension_disabled.path_offset
  assert command.path_angle == extension_disabled.path_angle


def test_single_preview_observation_cannot_pull_ordinary_c2_into_polynomial_transition():
  controller = ProjectedLatControlPath()
  curvature = 0.0017
  offset_curvature = 0.0047
  angle_curvature = 0.0025
  target = model(
    0.5 * offset_curvature * 7.0 ** 2,
    angle_curvature * 12.0,
    curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      target, curvature, 12.0, True, False,
      projected_measured_curvature=curvature,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert abs(command.curvature - curvature) < 1e-9


def test_low_speed_coherent_preview_acts_before_tracking_error_without_replacing_c2():
  controller = ProjectedLatControlPath()
  target = model(0.245, 0.07, 0.004, 0.0005)

  command = controller.update(
    target, 0.004, 5.0, True, False,
    projected_measured_curvature=0.004,
    desired_angle_curvature=0.004,
  )

  assert 0.0 < command.path_offset <= target.pathOffset
  assert 0.0 < command.path_angle <= target.pathAngle
  assert command.curvature == target.curvature


def test_coherent_preview_cannot_oppose_desired_steering_direction():
  controller = ProjectedLatControlPath()
  target = model(0.245, 0.07, 0.004, 0.0005)

  command = controller.update(
    target, -0.004, 5.0, True, False,
    projected_measured_curvature=-0.004,
    desired_angle_curvature=-0.004,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0


def test_ordinary_c2_is_immediate_but_sign_reversal_remains_bounded():
  controller = ProjectedLatControlPath()

  outward = controller.update(model(0.0, 0.0, 0.004), 0.0, 5.0, True, False)
  reversal = controller.update(model(0.0, 0.0, -0.004), 0.0, 5.0, True, False)

  assert outward.curvature == 0.004
  assert reversal.curvature == -0.0002


def test_meaningful_c3_preview_can_leave_c2_baseband():
  controller = ProjectedLatControlPath()
  target = model(0.0, 0.0, 0.001, 0.0005)

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.001, 25.0, True, False,
      projected_measured_curvature=0.001,
      desired_angle_curvature=0.001,
    )

  assert command is not None
  assert 0.0 < command.curvature < target.curvature
  assert command.curvature_rate > 0.0


def test_gentle_steady_curve_does_not_reset_when_wheel_temporarily_overtracks():
  controller = ProjectedLatControlPath()
  curvature = 0.003
  target = model(0.5 * curvature * 7.0 ** 2, curvature * 7.0, curvature)

  command = None
  for _ in range(20):
    command = controller.update(
      target, curvature, 7.0, True, False,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  settled_curvature = equivalent_curvature(command, 7.0)
  disturbed = controller.update(
    target, 0.0065, 7.0, True, False,
    desired_angle_curvature=curvature,
  )

  assert abs(settled_curvature - curvature) < 1e-9
  assert equivalent_curvature(disturbed, 7.0) >= 0.5 * settled_curvature
  assert disturbed.path_offset >= 0.0
  assert disturbed.path_angle >= 0.0
  assert disturbed.curvature > 0.0


def test_zero_unwind_target_cannot_manufacture_opposing_preview():
  controller = ProjectedLatControlPath()
  target = model(-0.0245, -0.007, -0.001)

  command = None
  for _ in range(10):
    command = controller.update(
      target, -0.0031, 7.0, True, False,
      desired_angle_curvature=-0.001,
    )

  assert command is not None
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_c3_unwind_waits_while_wheel_undertracks_desired_angle():
  controller = ProjectedLatControlPath()
  target = model(0.5, 0.1, 0.01, -0.0004)

  command = controller.update(
    target, 0.006, 7.0, True, False,
    projected_measured_curvature=0.006,
    desired_angle_curvature=0.01,
  )

  assert command.curvature_rate == 0.0


def test_gentle_c3_unwind_stops_after_projected_wheel_crosses_target():
  controller = ProjectedLatControlPath()
  target = model(-0.0245, -0.007, -0.00175, 0.00052)

  command = controller.update(
    target, -0.00175, 7.0, True, False,
    projected_measured_curvature=0.0005,
    desired_angle_curvature=-0.00175,
  )

  assert command.curvature_rate == 0.0


def test_projection_constraints_cannot_flip_delivered_model_geometry():
  controller = ProjectedLatControlPath()
  target = model(-0.1324408266, -0.0564401015, 0.0084251088, -0.0005080627)

  command = controller.update(
    target, -0.005, 10.0, True, False,
    projected_measured_curvature=-0.005,
    desired_angle_curvature=-0.005,
  )

  assert equivalent_curvature(target, 7.0) < -0.01
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_driver_override_projects_the_delivered_wheel_path():
  controller = ProjectedLatControlPath()

  command = controller.update(model(1.0, 0.2, 0.02, 0.001), -0.01, 10.0, True, True)

  assert command.path_offset == 0.5 * -0.01 * 7.0 ** 2
  assert command.path_angle == -0.01 * 10.0
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.0


def test_c0_c1_attack_uses_full_signal_range_and_release_is_immediate():
  controller = ProjectedLatControlPath()

  attack = controller.update(model(4.0, 0.4, 0.02, 0.001), 0.0, 7.0, True, False)
  release = controller.update(model(0.0, 0.0), 0.0, 7.0, True, False)

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497
  assert 0.0 <= attack.curvature_rate <= 0.0002
  assert release.path_offset == 0.0
  assert release.path_angle == 0.0
  assert release.curvature == 0.0
  assert release.curvature_rate == 0.0


def test_large_undertracking_maneuver_is_not_software_rate_limited():
  controller = ProjectedLatControlPath()
  target = model(4.0, 0.4, 0.02, 0.001)

  attack = controller.update(
    target, 0.0, 7.0, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.02,
  )

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497


def test_projected_arrival_does_not_reintroduce_c0_c1_attack_limit():
  controller = ProjectedLatControlPath()
  target = model(4.0, 0.4, 0.02, 0.001)

  attack = controller.update(
    target, 0.015, 7.0, True, False,
    projected_measured_curvature=0.021,
    desired_angle_curvature=0.02,
  )

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497


def test_measured_arrival_with_inward_spatial_slope_releases_only_outward_preview():
  desired_curvature = 0.015
  spatial_curvature = 0.04
  target = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    desired_curvature,
    -0.002,
  )
  behind_controller = ProjectedLatControlPath()
  arrived_controller = ProjectedLatControlPath()

  behind = None
  arrived = None
  for _ in range(100):
    behind = behind_controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.01,
      desired_angle_curvature=desired_curvature,
    )
    arrived = arrived_controller.update(
      target, 0.02, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_curvature,
    )

  assert behind is not None
  assert arrived is not None
  assert arrived.curvature == 0.0
  assert arrived.curvature_rate == -0.001024
  assert 0.0 <= arrived.path_offset < target.pathOffset
  assert 0.0 <= arrived.path_angle < target.pathAngle


def test_projected_arrival_cannot_release_preview_before_measured_wheel_arrives():
  desired_curvature = 0.015
  spatial_curvature = 0.04
  target = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    desired_curvature,
    -0.002,
  )
  measured_controller = ProjectedLatControlPath()
  projected_controller = ProjectedLatControlPath()

  measured = None
  projected = None
  for _ in range(100):
    measured = measured_controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.01,
      desired_angle_curvature=desired_curvature,
    )
    projected = projected_controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_curvature,
    )

  assert measured is not None
  assert projected is not None
  # This is lctnr's existing projected-wheel behavior. The new measured-wheel
  # release guard must not add another reduction while the actual wheel trails.
  assert projected.path_offset == target.pathOffset
  assert projected.path_angle == target.pathAngle
  assert projected.curvature == 0.0
  assert projected.curvature_rate == -0.001024


def test_measured_arrival_keeps_outward_preview_while_spatial_path_is_growing():
  desired_curvature = 0.015
  spatial_curvature = 0.04
  target = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    desired_curvature,
    0.002,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.02, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  # Preserve lctnr's existing C0 continuation correction while the spatial
  # polynomial is still building outward.
  assert abs(command.path_offset - 1.13925) < 1e-12
  assert command.path_angle == target.pathAngle
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.001023


def test_measured_arrival_bounds_known_stale_preview_without_touching_c2_c3():
  raw_coefficients = (4.669087, 0.716898, 0.004724, -0.017120)
  desired_curvature = 0.005703
  measured_curvature = 0.101705

  for direction in (1.0, -1.0):
    target = model(*(direction * value for value in raw_coefficients))
    controller = ProjectedLatControlPath()

    command = None
    for _ in range(100):
      command = controller.update(
        target, direction * measured_curvature, 2.719, True, False,
        projected_measured_curvature=direction * 0.101900,
        desired_angle_curvature=direction * desired_curvature,
      )

    assert command is not None
    assert 0.0 < direction * command.path_offset < abs(raw_coefficients[0])
    assert 0.0 < direction * command.path_angle < abs(raw_coefficients[1])
    assert command.curvature == 0.0
    assert command.curvature_rate == (-0.001024 if direction > 0.0 else 0.001023)
    assert abs(equivalent_curvature(command, 7.0)) <= desired_curvature + PATH_C0_CONTINUATION_MARGIN + 1e-12


def test_measured_arrival_release_does_not_delay_immediate_relatch():
  releasing_target = model(0.98, 0.28, 0.015, -0.002)
  arrived_controller = ProjectedLatControlPath()
  behind_controller = ProjectedLatControlPath()

  arrived = None
  behind = None
  for _ in range(100):
    arrived = arrived_controller.update(
      releasing_target, 0.020, 7.0, True, False,
      projected_measured_curvature=0.020,
      desired_angle_curvature=0.015,
    )
    behind = behind_controller.update(
      releasing_target, 0.010, 7.0, True, False,
      projected_measured_curvature=0.020,
      desired_angle_curvature=0.015,
    )

  assert arrived is not None
  assert behind is not None
  assert arrived.path_offset < behind.path_offset
  assert arrived.path_angle < behind.path_angle
  assert arrived.curvature == behind.curvature
  assert arrived.curvature_rate == behind.curvature_rate

  outward_target = model(1.47, 0.42, 0.030, 0.002)
  arrived_relatch = arrived_controller.update(
    outward_target, 0.010, 7.0, True, False,
    projected_measured_curvature=0.010,
    desired_angle_curvature=0.030,
  )
  behind_relatch = behind_controller.update(
    outward_target, 0.010, 7.0, True, False,
    projected_measured_curvature=0.010,
    desired_angle_curvature=0.030,
  )

  assert arrived_relatch == behind_relatch


def test_stale_preview_release_is_continuous_through_desired_zero():
  coefficients = (1.0, 0.2, 0.004, -0.001)
  outputs = [
    _taper_stale_outward_preview(coefficients, -0.003, desired_curvature, 0.020, 7.0)
    for desired_curvature in (1e-9, 0.0, -1e-9)
  ]

  for output in outputs:
    assert output[2:] == coefficients[2:]
    assert 0.0 <= equivalent_curvature(model(*output), 7.0) <= PATH_C0_CONTINUATION_MARGIN + 2e-9
  assert max(output[0] for output in outputs) - min(output[0] for output in outputs) < 1e-7
  assert max(output[1] for output in outputs) - min(output[1] for output in outputs) < 1e-7


def test_stale_preview_release_ignores_subthreshold_spatial_noise():
  coefficients = (1.0, 0.2, 0.004, -0.001)

  output = _taper_stale_outward_preview(
    coefficients, -0.001, 0.010, 0.020, 7.0,
  )

  assert output == coefficients


def test_ordinary_arrival_releases_stale_preview_with_outward_c3():
  raw_coefficients = (4.253234, 0.512123, 0.000270, 0.003431)
  desired_curvature = 0.000390
  measured_curvature = 0.034774
  ordinary_arrival_margin = 0.00025

  for direction in (1.0, -1.0):
    controller = ProjectedLatControlPath()
    target = model(*(direction * value for value in raw_coefficients))

    command = controller.update(
      target, direction * measured_curvature, 0.50, True, False,
      projected_measured_curvature=direction * 0.039742,
      desired_angle_curvature=direction * desired_curvature,
    )

    assert abs(command.path_offset) < abs(raw_coefficients[0])
    assert abs(command.path_angle) < abs(raw_coefficients[1])
    assert command.curvature == 0.0
    assert command.curvature_rate == direction * 0.0002
    assert abs(equivalent_curvature(command, 7.0)) <= desired_curvature + ordinary_arrival_margin + 1e-12


def test_ordinary_arrival_releases_preview_when_c3_is_zero_without_touching_c2():
  coefficients = (1.0, 0.2, 0.004, 0.0)

  output = _taper_stale_outward_preview(
    coefficients, 0.0, 0.001, 0.020, 7.0,
  )

  assert output[:2] != coefficients[:2]
  assert output[2:] == coefficients[2:]
  assert equivalent_curvature(model(*output), 7.0) <= coefficients[2] + 1e-12


def test_ordinary_arrival_release_relatches_immediately_when_desired_moves_outward():
  releasing_target = model(4.253234, 0.512123, 0.000270, 0.003431)
  arrived_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()

  arrived = arrived_controller.update(
    releasing_target, 0.034774, 0.50, True, False,
    projected_measured_curvature=0.039742,
    desired_angle_curvature=0.000390,
  )
  reference = reference_controller.update(
    releasing_target, 0.0002, 0.50, True, False,
    projected_measured_curvature=0.0002,
    desired_angle_curvature=0.000390,
  )

  assert arrived.path_offset < reference.path_offset
  assert arrived.path_angle < reference.path_angle
  assert arrived.curvature == reference.curvature
  assert arrived.curvature_rate == reference.curvature_rate

  outward_target = model(1.47, 0.42, 0.030, 0.002)
  arrived_relatch = arrived_controller.update(
    outward_target, 0.010, 7.0, True, False,
    projected_measured_curvature=0.010,
    desired_angle_curvature=0.030,
  )
  reference_relatch = reference_controller.update(
    outward_target, 0.010, 7.0, True, False,
    projected_measured_curvature=0.010,
    desired_angle_curvature=0.030,
  )

  assert arrived_relatch == reference_relatch
