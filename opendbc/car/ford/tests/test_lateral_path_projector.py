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


def test_strong_angle_rejection_follows_desired_angle_over_stale_model_path():
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

  assert equivalent_curvature(uncapped, 7.0) < 0.0
  assert equivalent_curvature(unwind, 7.0) > 0.0


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


def test_regrab_attack_bounds_continue_from_unscaled_base_command():
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.002, 0.0002)

  for _ in range(10):
    controller.update(
      target, 0.0025, 7.0, True, False,
      projected_measured_curvature=0.0025,
      desired_angle_curvature=0.0015,
    )
    reference_controller.update(
      target, 0.0025, 7.0, True, False,
      projected_measured_curvature=0.0025,
      desired_angle_curvature=0.01,
    )

  regrab = controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.01,
  )
  reference = reference_controller.update(
    target, 0.0025, 7.0, True, False,
    projected_measured_curvature=0.0025,
    desired_angle_curvature=0.01,
  )

  assert regrab == reference


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
  crossing_path = model(1.0, -1.0 / 7.0, -1e-8)

  command = controller.update(
    crossing_path, 0.002, 7.0, True, False,
    projected_measured_curvature=0.002,
    desired_angle_curvature=0.001,
  )

  assert -1e-7 < equivalent_curvature(crossing_path, 7.0) < 0.0
  assert 0.0 <= equivalent_curvature(command, 7.0) <= 0.001 + 1e-6


def test_target_crossing_is_continuous_in_both_directions():
  for direction in (-1.0, 1.0):
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    desired = direction * 0.003
    epsilon = direction * 1e-6
    commands = [
      ProjectedLatControlPath().update(
        target, desired + projected_offset, 7.0, True, False,
        projected_measured_curvature=desired,
        desired_angle_curvature=desired,
      )
      for projected_offset in (-epsilon, epsilon)
    ]

    assert abs(equivalent_curvature(commands[0], 7.0) - equivalent_curvature(commands[1], 7.0)) <= 1e-4
    scales = (4.6, 0.5, 0.02, 0.001)
    normalized_deltas = [
      abs(first - second) / scale
      for first, second, scale in zip(
        (commands[0].path_offset, commands[0].path_angle, commands[0].curvature, commands[0].curvature_rate),
        (commands[1].path_offset, commands[1].path_angle, commands[1].curvature, commands[1].curvature_rate),
        scales,
        strict=True,
      )
    ]
    assert max(normalized_deltas) <= 0.01


def test_projected_arrival_brakes_before_crossing_without_collapsing_authority():
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  target = model(0.4, 0.1, 0.002, 0.0002)
  deeper_target = model(0.8, 0.2, 0.002, 0.0002)
  desired = 0.003

  for _ in range(100):
    previous = controller.update(
      target, 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.01,
    )
    reference_controller.update(
      target, 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.01,
    )

  braked = controller.update(
    deeper_target, 0.002, 7.0, True, False,
    projected_measured_curvature=0.004,
    desired_angle_curvature=desired,
  )
  full = reference_controller.update(
    deeper_target, 0.002, 7.0, True, False,
    projected_measured_curvature=0.004,
    desired_angle_curvature=0.01,
  )

  assert equivalent_curvature(braked, 7.0) < equivalent_curvature(full, 7.0)
  assert equivalent_curvature(braked, 7.0) >= 0.89 * equivalent_curvature(previous, 7.0)


def test_constant_target_projection_jitter_cannot_form_an_authority_relay():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    desired = direction * 0.003
    epsilon = direction * 1e-6
    for _ in range(100):
      controller.update(
        target, desired, 7.0, True, False,
        projected_measured_curvature=desired - epsilon,
        desired_angle_curvature=desired,
      )

    commands = [
      controller.update(
        target, desired, 7.0, True, False,
        projected_measured_curvature=desired + (epsilon if i % 2 else -epsilon),
        desired_angle_curvature=desired,
      )
      for i in range(20)
    ]
    equivalent_curvatures = [direction * equivalent_curvature(command, 7.0) for command in commands]
    assert max(equivalent_curvatures) - min(equivalent_curvatures) <= 5e-4

    scales = (4.6, 0.5, 0.02, 0.001)
    for previous, current in zip(commands[:-1], commands[1:], strict=True):
      normalized_deltas = [
        abs(first - second) / scale
        for first, second, scale in zip(
          (previous.path_offset, previous.path_angle, previous.curvature, previous.curvature_rate),
          (current.path_offset, current.path_angle, current.curvature, current.curvature_rate),
          scales,
          strict=True,
        )
      ]
      assert max(normalized_deltas) <= 0.01


def test_constant_target_measured_bump_cannot_form_an_authority_relay():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    desired = direction * 0.003
    for _ in range(100):
      controller.update(
        target, desired, 7.0, True, False,
        projected_measured_curvature=desired,
        desired_angle_curvature=desired,
      )

    measured_curvatures = (
      desired + direction * 0.0001,
      desired - direction * 0.0012,
      desired + direction * 0.0001,
      desired - direction * 0.0012,
    )
    commands = [
      controller.update(
        target, measured, 7.0, True, False,
        projected_measured_curvature=desired,
        desired_angle_curvature=desired,
      )
      for measured in measured_curvatures
    ]
    delivered_curvatures = [direction * equivalent_curvature(command, 7.0) for command in commands]

    assert max(delivered_curvatures) <= 2.0 * direction * desired


def test_tiny_desired_noise_cannot_bypass_measured_relatch_confirmation():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    desired = direction * 0.003
    for _ in range(100):
      controller.update(
        target, desired, 7.0, True, False,
        projected_measured_curvature=desired,
        desired_angle_curvature=desired,
      )

    controller.update(
      target, desired + direction * 0.0001, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=desired,
    )
    noisy_desired = desired + direction * 1e-8
    command = controller.update(
      target, desired - direction * 0.0012, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=noisy_desired,
    )

    assert direction * equivalent_curvature(command, 7.0) <= 2.0 * direction * noisy_desired


def test_projected_undertracking_relatches_immediately():
  controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.002, 0.0002)
  desired = 0.003
  for _ in range(100):
    controller.update(
      target, desired, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=desired,
    )
    reference_controller.update(
      target, desired, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=0.01,
    )

  relatched = controller.update(
    target, desired - 0.0012, 7.0, True, False,
    projected_measured_curvature=desired - 0.0012,
    desired_angle_curvature=desired,
  )
  reference = reference_controller.update(
    target, desired - 0.0012, 7.0, True, False,
    projected_measured_curvature=desired - 0.0012,
    desired_angle_curvature=0.01,
  )

  assert relatched == reference


def test_safe_shallower_command_bypasses_relatch_confirmation():
  limited_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  deep_target = model(0.8, 0.2, 0.002, 0.0002)
  desired = 0.003
  for _ in range(100):
    limited_controller.update(
      deep_target, desired, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=desired,
    )
    reference_controller.update(
      deep_target, desired, 7.0, True, False,
      projected_measured_curvature=desired,
      desired_angle_curvature=desired,
    )

  safe_target = model(0.0, 0.0, 0.002)
  safe_release = limited_controller.update(
    safe_target, desired, 7.0, True, False,
    projected_measured_curvature=desired,
    desired_angle_curvature=desired,
  )
  untapered_reference = reference_controller.update(
    safe_target, desired, 7.0, True, False,
    projected_measured_curvature=desired,
  )

  assert safe_release == untapered_reference


def test_turn_in_preserves_37f_attack_speed_and_authority():
  expected_attack = (
    (0.147, 0.042, 0.0002, 0.0002),
    (0.294, 0.084, 0.0004, 0.0002),
    (0.441, 0.126, 0.0006, 0.0002),
    (0.588, 0.168, 0.0008, 0.0002),
    (0.735, 0.210, 0.0010, 0.0002),
  )
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    for expected in expected_attack:
      command = controller.update(
        target, direction * 0.002, 7.0, True, False,
        projected_measured_curvature=direction * 0.002,
        desired_angle_curvature=direction * 0.01,
      )
      assert all(
        abs(actual - direction * wanted) <= 1e-12
        for actual, wanted in zip(
          (command.path_offset, command.path_angle, command.curvature, command.curvature_rate),
          expected,
          strict=True,
        )
      )

    for _ in range(95):
      command = controller.update(
        target, direction * 0.002, 7.0, True, False,
        projected_measured_curvature=direction * 0.002,
        desired_angle_curvature=direction * 0.01,
      )
    assert all(
      abs(actual - direction * wanted) <= 1e-9
      for actual, wanted in zip(
        (command.path_offset, command.path_angle, command.curvature, command.curvature_rate),
        (0.8, 0.2, 0.002, 0.0002),
        strict=True,
      )
    )


def test_at_or_past_target_cannot_command_deeper():
  for direction in (-1.0, 1.0):
    target = model(direction * 0.8, direction * 0.2, direction * 0.002, direction * 0.0002)
    desired = direction * 0.003
    for past_target in (0.0, 1e-6, 1e-3):
      wheel_curvature = desired + direction * past_target
      command = ProjectedLatControlPath().update(
        target, wheel_curvature, 7.0, True, False,
        projected_measured_curvature=wheel_curvature,
        desired_angle_curvature=desired,
      )
      command_along = direction * equivalent_curvature(command, 7.0)
      assert 0.0 <= command_along <= direction * wheel_curvature + 1e-6


def test_arrival_brake_does_not_poison_the_37f_attack_plan():
  target = model(0.8, 0.2, 0.002, 0.0002)
  braked_controller = ProjectedLatControlPath()
  reference_controller = ProjectedLatControlPath()
  for _ in range(10):
    braked_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=0.003,
    )
    reference_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=0.01,
    )

  braked_regrab = braked_controller.update(
    target, 0.003, 7.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=0.01,
  )
  reference = reference_controller.update(
    target, 0.003, 7.0, True, False,
    projected_measured_curvature=0.003,
    desired_angle_curvature=0.01,
  )
  assert braked_regrab == reference


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


def test_strong_angle_rejection_can_countercommand_a_stale_mixed_model():
  controller = ProjectedLatControlPath()
  mixed_path = model(-1.815954395, 0.330586332, 0.018214388, 0.000487307)

  command = controller.update(
    mixed_path, -0.013445773, 10.0, True, False,
    projected_measured_curvature=-0.013445773,
    desired_angle_curvature=-0.019647983,
  )

  assert equivalent_curvature(mixed_path, 7.0) > 0.003
  assert equivalent_curvature(command, 7.0) < 0.0


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
