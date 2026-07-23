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


def test_projected_wheel_motion_cannot_scale_c0_c1_authority():
  target = model(0.5, 0.1, 0.01)
  measured_controller = ProjectedLatControlPath()
  projected_controller = ProjectedLatControlPath()

  measured_command = measured_controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.005,
    desired_angle_curvature=0.01,
  )
  projected_command = projected_controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.03,
    desired_angle_curvature=0.01,
  )

  assert measured_command.path_offset == projected_command.path_offset
  assert measured_command.path_angle == projected_command.path_angle


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


def test_c3_unwind_waits_while_wheel_undertracks_desired_angle():
  controller = ProjectedLatControlPath()
  target = model(0.5, 0.1, 0.01, -0.0004)

  command = controller.update(
    target, 0.006, 7.0, True, False,
    projected_measured_curvature=0.006,
    desired_angle_curvature=0.01,
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
