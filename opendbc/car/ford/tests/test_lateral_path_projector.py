from types import SimpleNamespace

from opendbc.car.ford.lateral_path_projector import ProjectedLatControlPath


def model(path_offset: float, path_angle: float, curvature: float = 0.0, curvature_rate: float = 0.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=path_offset,
    pathAngle=path_angle,
    curvature=curvature,
    curvatureRate=curvature_rate,
  )


def equivalent_curvature(command, distance: float) -> float:
  path_offset = getattr(command, "path_offset", getattr(command, "pathOffset", 0.0))
  path_angle = getattr(command, "path_angle", getattr(command, "pathAngle", 0.0))
  curvature_rate = getattr(command, "curvature_rate", getattr(command, "curvatureRate", 0.0))
  y = path_offset + path_angle * distance + 0.5 * command.curvature * distance ** 2 + \
      curvature_rate * distance ** 3 / 6.0
  return 2.0 * y / distance ** 2


def test_feasible_strong_model_is_reproduced_once_wheel_delivers_it():
  controller = ProjectedLatControlPath()
  delivered_curvature = 0.015
  target = model(
    0.5 * delivered_curvature * 7.0 ** 2,
    delivered_curvature * 7.0,
  )

  command = None
  for _ in range(100):
    command = controller.update(target, delivered_curvature, 7.0, True, False)

  assert command is not None
  for distance in (3.0, 7.0, 15.0, 30.0):
    assert abs(equivalent_curvature(command, distance) - equivalent_curvature(target, distance)) < 1e-6


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


def test_desired_angle_shortfall_extends_c0_c1_beyond_model_geometry():
  model_curvature = 0.008
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    model_curvature,
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
  continuing_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, 0.0004)

  command = controller.update(
    continuing_geometry, 0.01, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) > 0.0


def test_opposing_action_does_not_discard_model_preview_before_wheel_follows_it():
  controller = ProjectedLatControlPath()
  entering_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004)

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
  target = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0)
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


def test_transition_crossfades_once_from_c2_to_full_polynomial():
  commands = []
  for curvature in (0.003, 0.0045, 0.006):
    controller = ProjectedLatControlPath()
    target = model(
      0.5 * curvature * 7.0 ** 2,
      curvature * 7.0,
      curvature,
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
  assert gentle.curvature == 0.003
  assert gentle.path_offset == 0.0
  assert gentle.path_angle == 0.0
  assert abs(transition.curvature - 0.00225) < 1e-9
  assert transition.path_offset > 0.0
  assert transition.path_angle > 0.0
  assert full.curvature == 0.0
  assert full.path_offset > transition.path_offset
  assert full.path_angle > transition.path_angle


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


def test_attack_is_bounded_and_release_is_immediate():
  controller = ProjectedLatControlPath()

  attack = controller.update(model(4.0, 0.4, 0.02, 0.001), 0.0, 7.0, True, False)
  release = controller.update(model(0.0, 0.0), 0.0, 7.0, True, False)

  assert 0.0 < attack.path_offset <= 0.147
  assert 0.0 < attack.path_angle <= 0.042
  assert 0.0 <= attack.curvature_rate <= 0.0002
  assert release.path_offset == 0.0
  assert release.path_angle == 0.0
  assert release.curvature == 0.0
  assert release.curvature_rate == 0.0
