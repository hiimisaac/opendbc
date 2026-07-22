import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_path import (
  LatControlPath,
  SteeringAngleProjector,
  driver_steering_opposes_command,
  projected_tracking_error,
)


def polynomial_model(curvature: float, curvature_rate: float = 0.0, lookahead: float = 7.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=0.5 * curvature * 7.0 ** 2 + curvature_rate * 7.0 ** 3 / 6.0,
    pathAngle=curvature * lookahead + 0.5 * curvature_rate * lookahead ** 2,
    curvatureRate=curvature_rate,
  )


def split_geometry_model(offset_curvature: float, angle_curvature: float):
  return SimpleNamespace(
    valid=True,
    pathOffset=0.5 * offset_curvature * 7.0 ** 2,
    pathAngle=angle_curvature * 7.0,
    curvatureRate=0.0,
  )


def with_action(path, curvature: float):
  if path is None:
    path = SimpleNamespace(valid=False, pathOffset=0.0, pathAngle=0.0, curvatureRate=0.0)
  path.curvature = curvature
  return path


def test_projected_tracking_error_only_discounts_closing_wheel_motion():
  assert math.isclose(projected_tracking_error(0.04, 0.02, 0.03), 0.01)
  assert projected_tracking_error(0.04, 0.02, 0.05) == 0.0
  assert math.isclose(projected_tracking_error(0.04, 0.02, 0.01), 0.02)


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


def test_model_geometry_retains_path_authority_missing_from_action():
  controller = LatControlPath()

  first = controller.update(
    path=with_action(polynomial_model(0.01, 0.0004), 0.0034),
    measured_curvature=0.0,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )
  command = controller.update(
    path=with_action(polynomial_model(0.01, 0.0004), 0.0034),
    measured_curvature=0.0,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.path_offset > 0.1
  assert command.path_angle > 0.05
  assert math.isclose(first.curvature, 0.0002)


def test_falling_action_does_not_unwind_while_model_still_requires_turn():
  controller = LatControlPath()

  for _ in range(10):
    controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)
  command = controller.update(with_action(polynomial_model(0.015), 0.0), 0.015, 7.0, True, False)

  assert command.path_offset >= 0.0
  assert command.path_angle > 0.0


def test_model_relative_undertracking_adds_bounded_path_correction():
  controller = LatControlPath()

  command = None
  for _ in range(20):
    command = controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)

  assert command is not None
  assert command.path_offset > 0.5 * 0.015 * 7.0 ** 2
  assert command.path_angle > 0.015 * 7.0


def test_c0_and_c1_preserve_independent_model_geometry():
  stronger_c0_controller = LatControlPath()
  matched_geometry_controller = LatControlPath()

  for _ in range(20):
    stronger_c0 = stronger_c0_controller.update(
      with_action(split_geometry_model(0.03, 0.015), 0.012), 0.0, 7.0, True, False,
    )
    matched_geometry = matched_geometry_controller.update(
      with_action(split_geometry_model(0.015, 0.015), 0.012), 0.0, 7.0, True, False,
    )

  assert stronger_c0.path_offset > matched_geometry.path_offset
  assert abs(stronger_c0.path_angle - matched_geometry.path_angle) < 0.001


def test_desired_angle_conflict_continuously_unwinds_stale_preview():
  controller = LatControlPath()

  controller.update(
    with_action(polynomial_model(-0.004), -0.0009), -0.013, 3.0, True, False,
    projected_measured_curvature=-0.013,
    desired_angle_curvature=0.0002,
  )
  command = controller.update(
    with_action(polynomial_model(-0.08), 0.0001), -0.027, 3.0, True, False,
    projected_measured_curvature=-0.027,
    desired_angle_curvature=0.0002,
  )

  assert command.path_offset > 0.0
  assert command.path_angle > 0.0


def test_low_speed_preview_is_not_mistaken_for_stale_release():
  baseline_controller = LatControlPath()
  angle_feedback_controller = LatControlPath()

  for controller in (baseline_controller, angle_feedback_controller):
    controller.update(
      with_action(polynomial_model(0.004), 0.0001), 0.055, 1.5, True, False,
      projected_measured_curvature=0.055,
      desired_angle_curvature=0.0 if controller is angle_feedback_controller else None,
    )

  baseline = baseline_controller.update(
    with_action(polynomial_model(0.08), 0.0001), 0.055, 1.5, True, False,
    projected_measured_curvature=0.055,
  )
  angle_feedback = angle_feedback_controller.update(
    with_action(polynomial_model(0.08), 0.0001), 0.055, 1.5, True, False,
    projected_measured_curvature=0.055,
    desired_angle_curvature=0.0,
  )

  assert angle_feedback == baseline


def test_small_desired_angle_error_does_not_weaken_model_preview():
  baseline_controller = LatControlPath()
  angle_feedback_controller = LatControlPath()

  for _ in range(20):
    baseline = baseline_controller.update(
      with_action(polynomial_model(0.08), 0.06), 0.065, 5.0, True, False,
      projected_measured_curvature=0.065,
      desired_angle_curvature=0.065,
    )
    angle_feedback = angle_feedback_controller.update(
      with_action(polynomial_model(0.08), 0.06), 0.065, 5.0, True, False,
      projected_measured_curvature=0.065,
      desired_angle_curvature=0.063,
    )

  assert math.isclose(angle_feedback.path_offset, baseline.path_offset)
  assert math.isclose(angle_feedback.path_angle, baseline.path_angle)


def test_projected_wheel_motion_reduces_only_added_c0_tracking_correction():
  lagging_controller = LatControlPath()
  closing_controller = LatControlPath()

  for _ in range(10):
    lagging = lagging_controller.update(
      with_action(polynomial_model(0.015), 0.012), 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
    )
    closing = closing_controller.update(
      with_action(polynomial_model(0.015), 0.012), 0.002, 7.0, True, False,
      projected_measured_curvature=0.010,
    )

  assert 0.0 < closing.path_offset < lagging.path_offset
  assert math.isclose(closing.path_angle, lagging.path_angle)


def test_projected_wheel_motion_preserves_c0_feedback_floor():
  controller = LatControlPath()
  conservative_controller = LatControlPath()

  for _ in range(10):
    command = controller.update(
      with_action(polynomial_model(0.015), 0.012), 0.002, 7.0, True, False,
      projected_measured_curvature=0.020,
    )
    conservative = conservative_controller.update(
      with_action(polynomial_model(0.015), 0.012), 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
    )
  assert command.path_offset > 0.0
  assert command.path_offset < conservative.path_offset
  assert math.isclose(command.path_angle, conservative.path_angle)


def test_tiny_action_sign_noise_cannot_promote_model_to_full_c0_reversal():
  controller = LatControlPath()

  command = None
  for _ in range(20):
    command = controller.update(with_action(polynomial_model(0.09), -0.0001), 0.05, 7.0, True, False)

  assert command is not None
  assert 0.0 < command.path_offset <= 0.5 * 0.012 * 7.0 ** 2
  assert command.path_angle > 0.0


def test_action_unwind_releases_stale_preview_without_relatching():
  controller = LatControlPath()

  attack = controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)
  release = controller.update(with_action(polynomial_model(0.001), 0.0), 0.015, 7.0, True, False)
  unwind = controller.update(with_action(polynomial_model(0.001), 0.0), 0.015, 7.0, True, False)
  continued_unwind = controller.update(with_action(polynomial_model(0.001), 0.0), 0.015, 7.0, True, False)

  assert attack.path_offset > 0.0
  assert attack.path_angle > 0.0
  assert release.path_offset < 0.0
  assert release.path_angle < 0.0
  assert unwind.path_offset < 0.0
  assert unwind.path_angle < 0.0
  assert continued_unwind.path_offset <= unwind.path_offset
  assert continued_unwind.path_angle <= unwind.path_angle


def test_small_same_direction_action_cannot_preserve_preview_while_unwinding():
  controller = LatControlPath()

  command = controller.update(with_action(polynomial_model(0.001), 0.00275), 0.015, 7.0, True, False)

  assert command.path_offset <= 0.0
  assert command.path_angle <= 0.0


def test_unwind_has_no_relatch_discontinuity_above_straightening_threshold():
  controller = LatControlPath()

  command = controller.update(with_action(polynomial_model(0.001), 0.0031), 0.015, 7.0, True, False)

  assert command.path_offset <= 0.0
  assert command.path_angle <= 0.0


def test_action_bounce_cannot_relatch_after_unwind_begins():
  controller = LatControlPath()

  controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)
  unwind = controller.update(with_action(polynomial_model(0.001), 0.0), 0.015, 7.0, True, False)
  bounced_action = controller.update(with_action(polynomial_model(0.001), 0.0045), 0.015, 7.0, True, False)

  assert unwind.path_offset < 0.0
  assert unwind.path_angle < 0.0
  assert bounced_action.path_offset < 0.0
  assert bounced_action.path_angle < 0.0


def test_action_drop_starts_unwind_without_a_straight_intermediate_frame():
  controller = LatControlPath()

  controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)
  direct_unwind = controller.update(with_action(polynomial_model(0.001), 0.0045), 0.015, 7.0, True, False)

  assert direct_unwind.path_offset < 0.0
  assert direct_unwind.path_angle < 0.0


def test_unwind_releases_when_action_genuinely_moves_beyond_the_wheel():
  controller = LatControlPath()

  controller.update(with_action(polynomial_model(0.015), 0.012), 0.0, 7.0, True, False)
  controller.update(with_action(polynomial_model(0.001), 0.0), 0.015, 7.0, True, False)
  renewed_turn = controller.update(with_action(polynomial_model(0.020), 0.020), 0.015, 7.0, True, False)

  assert renewed_turn.path_offset > 0.0
  assert renewed_turn.path_angle > 0.0


def test_model_relative_correction_fades_as_wheel_reaches_path():
  def angle_equivalent(measured_curvature: float) -> float:
    controller = LatControlPath()
    command = None
    for _ in range(40):
      command = controller.update(with_action(polynomial_model(0.015), 0.0045), measured_curvature, 7.0, True, False)
    assert command is not None
    return command.curvature + command.path_angle / 7.0

  far_from_target = angle_equivalent(0.0)
  near_target = angle_equivalent(0.010)
  at_target = angle_equivalent(0.015)

  assert far_from_target > near_target
  assert near_target > at_target
  assert at_target > 0.0045


def test_moving_turn_retains_preview_authority_as_speed_and_lookahead_drop():
  controller = LatControlPath()

  at_25_mph = controller.update(with_action(polynomial_model(0.015, lookahead=11.2), 0.008), 0.004, 11.2, True, False)
  at_15_mph = controller.update(with_action(polynomial_model(0.015), 0.010), 0.006, 6.7, True, False)

  assert at_25_mph.path_offset > 0.0
  assert at_25_mph.path_angle > 0.0
  assert at_15_mph.path_offset >= at_25_mph.path_offset
  assert at_15_mph.path_angle > 0.0


def test_spatial_slope_stays_quiet_when_action_and_vehicle_already_turn_together():
  controller = LatControlPath()

  command = controller.update(
    path=with_action(polynomial_model(0.01, 0.0004), 0.0034),
    measured_curvature=0.01,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.curvature_rate == 0.0


def test_spatial_slope_leads_a_coherent_moving_reversal():
  controller = LatControlPath()

  command = controller.update(
    path=with_action(polynomial_model(-0.008, -0.0004), -0.0034),
    measured_curvature=0.004,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.curvature_rate < -0.0001


def test_growing_path_authority_is_bounded_but_release_is_immediate():
  controller = LatControlPath()

  attack = controller.update(with_action(polynomial_model(0.05, 0.001), 0.02), 0.0, 7.0, True, False)
  release = controller.update(with_action(polynomial_model(0.0), 0.0), 0.0, 7.0, True, False)

  assert 0.0 < attack.path_offset <= 0.5 * 0.006 * 7.0 ** 2
  assert 0.0 < attack.path_angle <= 0.006 * 7.0
  assert 0.0 < attack.curvature_rate <= 0.0002
  assert release.path_offset == 0.0
  assert release.path_angle == 0.0
  assert release.curvature_rate == 0.0


def test_delivered_gentle_model_path_uses_smooth_curvature_alone():
  controller = LatControlPath()

  command = controller.update(
    path=with_action(polynomial_model(0.003, lookahead=15.0), 0.003),
    measured_curvature=0.003,
    v_ego=15.0,
    active=True,
    driver_override=False,
  )

  assert math.isclose(command.path_offset, 0.0, abs_tol=1e-4)
  assert math.isclose(command.path_angle, 0.0, abs_tol=1e-4)
  assert math.isclose(command.curvature, 0.0002)
  assert command.curvature_rate == 0.0


def test_low_amplitude_model_geometry_enters_without_a_heading_step():
  controller = LatControlPath()

  below_threshold = controller.update(
    path=with_action(polynomial_model(0.0029, lookahead=15.0), 0.0),
    measured_curvature=0.0,
    v_ego=15.0,
    active=True,
    driver_override=False,
  )
  above_threshold = controller.update(
    path=with_action(polynomial_model(0.0031, lookahead=15.0), 0.0),
    measured_curvature=0.0,
    v_ego=15.0,
    active=True,
    driver_override=False,
  )

  assert below_threshold.path_angle == 0.0
  assert 0.0 < above_threshold.path_angle < 0.01


def test_strong_path_placement_is_not_weakened_by_a_small_heading_component():
  controller = LatControlPath()
  reference_controller = LatControlPath()

  command = None
  reference = None
  for _ in range(20):
    command = controller.update(
      path=with_action(split_geometry_model(0.03, 0.0031), 0.012),
      measured_curvature=0.0,
      v_ego=7.0,
      active=True,
      driver_override=False,
    )
    reference = reference_controller.update(
      path=with_action(split_geometry_model(0.03, 0.006), 0.012),
      measured_curvature=0.0,
      v_ego=7.0,
      active=True,
      driver_override=False,
    )

  assert command is not None
  assert reference is not None
  assert math.isclose(command.path_offset, reference.path_offset)


def test_conditioning_cannot_reverse_coherent_geometry_toward_the_action():
  controller = LatControlPath()

  command = controller.update(
    path=with_action(polynomial_model(-0.0031), 0.0032),
    measured_curvature=-0.0007,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.path_offset < -0.01
  assert command.path_angle < -0.04


def test_large_turn_keeps_c2_flushed_until_the_wheel_settles():
  controller = LatControlPath()

  controller.update(with_action(polynomial_model(0.015), 0.015), 0.012, 7.0, True, False)
  early_exit = controller.update(with_action(polynomial_model(0.004), 0.004), 0.010, 7.0, True, False)
  settled = controller.update(with_action(polynomial_model(0.001), 0.001), 0.001, 7.0, True, False)

  assert early_exit.curvature == 0.0
  assert math.isclose(settled.curvature, 0.0002)


def test_inactive_resets_the_previous_command():
  controller = LatControlPath()
  controller.update(with_action(polynomial_model(0.015), 0.015), 0.012, 7.0, True, False)

  controller.update(with_action(polynomial_model(0.0), 0.0), 0.0, 7.0, False, False)
  reengaged = controller.update(with_action(polynomial_model(0.001), 0.001), 0.001, 7.0, True, False)

  assert math.isclose(reengaged.curvature, 0.0002)


def test_missing_model_marks_path_invalid_without_inventing_geometry():
  controller = LatControlPath()

  command = controller.update(with_action(None, 0.003), 0.0, 15.0, True, False)

  assert not command.valid
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature_rate == 0.0


def test_nonfinite_path_target_is_invalid():
  controller = LatControlPath()
  path = polynomial_model(0.01)
  path.pathOffset = float("nan")

  command = controller.update(with_action(path, 0.01), 0.0, 7.0, True, False)

  assert not command.valid
