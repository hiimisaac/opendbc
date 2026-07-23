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


def horizon_error(command, target) -> float:
  return sum(
    (equivalent_curvature(command, distance) - equivalent_curvature(target, distance)) ** 2
    for distance in (3.0, 7.0, 15.0, 30.0)
  )


def test_feasible_strong_model_is_reproduced_after_attack_settles():
  controller = ProjectedLatControlPath()
  target = model(0.5, 0.1)

  command = None
  for _ in range(100):
    command = controller.update(target, 0.0, 7.0, True, False)

  assert command is not None
  for distance in (3.0, 7.0, 15.0, 30.0):
    assert abs(equivalent_curvature(command, distance) - equivalent_curvature(target, distance)) < 1e-6


def test_clipped_coefficients_redistribute_residual_across_free_coefficients():
  controller = ProjectedLatControlPath()
  target = model(-2.7, -0.60, -0.05, -0.002)

  command = None
  for _ in range(100):
    command = controller.update(target, -0.03, 7.0, True, False)

  naive_clip = model(-2.7, -0.475, -0.02, -0.001024)
  assert command is not None
  assert horizon_error(command, target) < horizon_error(naive_clip, target)


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


def test_spatial_unwind_reduces_overtracking_without_reversing_model_path():
  uncapped_controller = ProjectedLatControlPath()
  unwind_controller = ProjectedLatControlPath()
  stale_preview = model(-1.96, -0.56, 0.0001)

  uncapped = uncapped_controller.update(
    stale_preview, -0.027, 3.0, True, False,
    projected_measured_curvature=-0.027,
    desired_angle_curvature=-0.027,
  )
  unwind = unwind_controller.update(
    stale_preview, -0.027, 3.0, True, False,
    projected_measured_curvature=-0.027,
    desired_angle_curvature=0.0002,
  )

  assert equivalent_curvature(stale_preview, 7.0) * equivalent_curvature(unwind, 7.0) > 0.0
  assert abs(equivalent_curvature(unwind, 7.0)) < abs(equivalent_curvature(uncapped, 7.0))
  assert abs(unwind.path_offset / unwind.path_angle - uncapped.path_offset / uncapped.path_angle) < 1e-6


def test_overtracked_wheel_cannot_be_commanded_deeper_without_changing_path_shape():
  uncapped_controller = ProjectedLatControlPath()
  capped_controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.002, 0.0002)

  uncapped = uncapped_controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.003,
  )
  capped = capped_controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.0015,
  )

  assert equivalent_curvature(capped, 7.0) <= 0.0025 + 1e-6
  coefficient_shares = [
    capped_coefficient / uncapped_coefficient
    for capped_coefficient, uncapped_coefficient in zip(
      (capped.path_offset, capped.path_angle, capped.curvature, capped.curvature_rate),
      (uncapped.path_offset, uncapped.path_angle, uncapped.curvature, uncapped.curvature_rate),
      strict=True,
    )
  ]
  assert max(coefficient_shares) - min(coefficient_shares) < 1e-9


def test_regrab_attack_bounds_start_from_delivered_scaled_command():
  controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.002, 0.0002)

  capped = controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.0015,
  )
  regrab = controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.003,
  )

  assert regrab.path_offset <= capped.path_offset + 0.147 + 1e-9
  assert regrab.path_angle <= capped.path_angle + 0.042 + 1e-9
  assert regrab.curvature <= capped.curvature + 0.0002 + 1e-9
  assert regrab.curvature_rate <= capped.curvature_rate + 0.0002 + 1e-9


def test_overtracking_ceiling_applies_while_spatial_path_is_still_deepening():
  controller = ProjectedLatControlPath()
  deepening_path = model(0.0, 0.0, 0.004, 0.0006)

  command = None
  for _ in range(100):
    command = controller.update(
      deepening_path, 0.004, 7.0, True, False,
      projected_measured_curvature=0.004,
      desired_angle_curvature=0.003,
    )

  assert command is not None
  assert equivalent_curvature(deepening_path, 15.0) > equivalent_curvature(deepening_path, 3.0)
  assert equivalent_curvature(command, 7.0) <= 0.004 + 1e-6


def test_overtracking_ceiling_also_applies_below_maneuver_curvature():
  controller = ProjectedLatControlPath()
  gentle_path = model(0.0, 0.0, 0.002)

  command = None
  for _ in range(100):
    command = controller.update(
      gentle_path, 0.0015, 7.0, True, False,
      projected_measured_curvature=0.0015,
      desired_angle_curvature=0.001,
    )

  assert command is not None
  assert equivalent_curvature(command, 7.0) <= 0.0015 + 1e-6


def test_crossing_model_uses_desired_direction_for_overtracking_ceiling():
  controller = ProjectedLatControlPath()
  crossing_path = model(1.0, -1.0 / 7.0)

  command = controller.update(
    crossing_path, 0.002, 7.0, True, False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.001,
  )

  assert abs(equivalent_curvature(crossing_path, 7.0)) < 1e-12
  assert equivalent_curvature(command, 7.0) <= 0.002 + 1e-6


def test_turn_in_authority_is_unchanged_before_wheel_reaches_desired_angle():
  reference_controller = ProjectedLatControlPath()
  turn_in_controller = ProjectedLatControlPath()
  target = model(0.8, 0.2)

  reference = reference_controller.update(
    target, 0.002, 7.0, True, False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.002,
  )
  turn_in = turn_in_controller.update(
    target, 0.002, 7.0, True, False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.01,
  )

  assert turn_in == reference


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


def test_attack_limited_mixed_model_cannot_countercommand_meaningful_path():
  controller = ProjectedLatControlPath()
  mixed_path = model(-1.815954395, 0.330586332, 0.018214388, 0.000487307)

  command = controller.update(
    mixed_path, -0.013445773, 10.0, True, False,
    projected_measured_curvature=-0.013445773,
    desired_angle_curvature=-0.019647983,
  )

  assert equivalent_curvature(mixed_path, 7.0) > 0.003
  assert equivalent_curvature(command, 7.0) >= 0.0


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
