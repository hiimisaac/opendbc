import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_path_shadow import LatControlPath


def polynomial_model(curvature: float, curvature_rate: float = 0.0):
  distances = (0.0, 1.75, 2.5, 3.5, 5.0, 7.0, 12.0, 20.0)
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * curvature * s ** 2 + curvature_rate * s ** 3 / 6.0 for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[curvature * s + 0.5 * curvature_rate * s ** 2 for s in distances],
    ),
  )


def split_geometry_model(offset_curvature: float, angle_curvature: float):
  distances = (0.0, 1.75, 2.5, 3.5, 5.0, 7.0, 12.0, 20.0)
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * offset_curvature * s ** 2 for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[angle_curvature * s for s in distances],
    ),
  )


def test_model_geometry_retains_path_authority_missing_from_action():
  controller = LatControlPath()

  first = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.0034,
    measured_curvature=0.0,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )
  command = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.0034,
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
    controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)
  command = controller.update(polynomial_model(0.015), 0.0, 0.015, 7.0, True, False)

  assert command.path_offset >= 0.0
  assert command.path_angle > 0.0


def test_model_relative_undertracking_adds_bounded_path_correction():
  controller = LatControlPath()

  command = None
  for _ in range(20):
    command = controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)

  assert command is not None
  assert command.path_offset > 0.5 * 0.015 * 7.0 ** 2
  assert command.path_angle > 0.015 * 7.0


def test_c0_and_c1_preserve_independent_model_geometry():
  stronger_c0_controller = LatControlPath()
  matched_geometry_controller = LatControlPath()

  for _ in range(20):
    stronger_c0 = stronger_c0_controller.update(
      split_geometry_model(0.03, 0.015), 0.012, 0.0, 7.0, True, False,
    )
    matched_geometry = matched_geometry_controller.update(
      split_geometry_model(0.015, 0.015), 0.012, 0.0, 7.0, True, False,
    )

  assert stronger_c0.path_offset > matched_geometry.path_offset
  assert abs(stronger_c0.path_angle - matched_geometry.path_angle) < 0.001


def test_desired_angle_conflict_continuously_unwinds_stale_preview():
  controller = LatControlPath()

  controller.update(
    polynomial_model(-0.004), -0.0009, -0.013, 3.0, True, False,
    projected_measured_curvature=-0.013,
    desired_angle_curvature=0.0002,
  )
  command = controller.update(
    polynomial_model(-0.08), 0.0001, -0.027, 3.0, True, False,
    projected_measured_curvature=-0.027,
    desired_angle_curvature=0.0002,
  )

  assert command.path_offset > 0.0
  assert command.path_angle > 0.0
  assert controller.preview_conflict_share > 0.0


def test_low_speed_preview_is_not_mistaken_for_stale_release():
  baseline_controller = LatControlPath()
  angle_feedback_controller = LatControlPath()

  for controller in (baseline_controller, angle_feedback_controller):
    controller.update(
      polynomial_model(0.004), 0.0001, 0.055, 1.5, True, False,
      projected_measured_curvature=0.055,
      desired_angle_curvature=0.0 if controller is angle_feedback_controller else None,
    )

  baseline = baseline_controller.update(
    polynomial_model(0.08), 0.0001, 0.055, 1.5, True, False,
    projected_measured_curvature=0.055,
  )
  angle_feedback = angle_feedback_controller.update(
    polynomial_model(0.08), 0.0001, 0.055, 1.5, True, False,
    projected_measured_curvature=0.055,
    desired_angle_curvature=0.0,
  )

  assert angle_feedback == baseline
  assert angle_feedback_controller.preview_conflict_share == 0.0


def test_small_desired_angle_error_does_not_weaken_model_preview():
  baseline_controller = LatControlPath()
  angle_feedback_controller = LatControlPath()

  for _ in range(20):
    baseline = baseline_controller.update(
      polynomial_model(0.08), 0.06, 0.065, 5.0, True, False,
      projected_measured_curvature=0.065,
      desired_angle_curvature=0.065,
    )
    angle_feedback = angle_feedback_controller.update(
      polynomial_model(0.08), 0.06, 0.065, 5.0, True, False,
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
      polynomial_model(0.015), 0.012, 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
    )
    closing = closing_controller.update(
      polynomial_model(0.015), 0.012, 0.002, 7.0, True, False,
      projected_measured_curvature=0.010,
    )

  assert 0.0 < closing.path_offset < lagging.path_offset
  assert math.isclose(closing.path_angle, lagging.path_angle)


def test_projected_wheel_motion_preserves_c0_feedback_floor():
  controller = LatControlPath()
  conservative_controller = LatControlPath()

  for _ in range(10):
    command = controller.update(
      polynomial_model(0.015), 0.012, 0.002, 7.0, True, False,
      projected_measured_curvature=0.020,
    )
    conservative = conservative_controller.update(
      polynomial_model(0.015), 0.012, 0.002, 7.0, True, False,
      projected_measured_curvature=0.002,
    )
  assert command.path_offset > 0.0
  assert command.path_offset < conservative.path_offset
  assert math.isclose(command.path_angle, conservative.path_angle)


def test_tiny_action_sign_noise_cannot_promote_model_to_full_c0_reversal():
  controller = LatControlPath()

  command = None
  for _ in range(20):
    command = controller.update(polynomial_model(0.09), -0.0001, 0.05, 7.0, True, False)

  assert command is not None
  assert 0.0 < command.path_offset <= 0.5 * 0.012 * 7.0 ** 2
  assert command.path_angle > 0.0


def test_action_unwind_releases_stale_preview_without_relatching():
  controller = LatControlPath()

  attack = controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)
  release = controller.update(polynomial_model(0.001), 0.0, 0.015, 7.0, True, False)
  unwind = controller.update(polynomial_model(0.001), 0.0, 0.015, 7.0, True, False)
  continued_unwind = controller.update(polynomial_model(0.001), 0.0, 0.015, 7.0, True, False)

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

  command = controller.update(polynomial_model(0.001), 0.00275, 0.015, 7.0, True, False)

  assert command.path_offset <= 0.0
  assert command.path_angle <= 0.0


def test_unwind_has_no_relatch_discontinuity_above_straightening_threshold():
  controller = LatControlPath()

  command = controller.update(polynomial_model(0.001), 0.0031, 0.015, 7.0, True, False)

  assert command.path_offset <= 0.0
  assert command.path_angle <= 0.0


def test_action_bounce_cannot_relatch_after_unwind_begins():
  controller = LatControlPath()

  controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)
  unwind = controller.update(polynomial_model(0.001), 0.0, 0.015, 7.0, True, False)
  bounced_action = controller.update(polynomial_model(0.001), 0.0045, 0.015, 7.0, True, False)

  assert unwind.path_offset < 0.0
  assert unwind.path_angle < 0.0
  assert bounced_action.path_offset < 0.0
  assert bounced_action.path_angle < 0.0


def test_action_drop_starts_unwind_without_a_straight_intermediate_frame():
  controller = LatControlPath()

  controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)
  direct_unwind = controller.update(polynomial_model(0.001), 0.0045, 0.015, 7.0, True, False)

  assert direct_unwind.path_offset < 0.0
  assert direct_unwind.path_angle < 0.0


def test_unwind_releases_when_action_genuinely_moves_beyond_the_wheel():
  controller = LatControlPath()

  controller.update(polynomial_model(0.015), 0.012, 0.0, 7.0, True, False)
  controller.update(polynomial_model(0.001), 0.0, 0.015, 7.0, True, False)
  renewed_turn = controller.update(polynomial_model(0.020), 0.020, 0.015, 7.0, True, False)

  assert renewed_turn.path_offset > 0.0
  assert renewed_turn.path_angle > 0.0


def test_model_relative_correction_fades_as_wheel_reaches_path():
  def angle_equivalent(measured_curvature: float) -> float:
    controller = LatControlPath()
    command = None
    for _ in range(40):
      command = controller.update(polynomial_model(0.015), 0.0045, measured_curvature, 7.0, True, False)
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

  at_25_mph = controller.update(polynomial_model(0.015), 0.008, 0.004, 11.2, True, False)
  at_15_mph = controller.update(polynomial_model(0.015), 0.010, 0.006, 6.7, True, False)

  assert at_25_mph.path_offset > 0.0
  assert at_25_mph.path_angle > 0.0
  assert at_15_mph.path_offset >= at_25_mph.path_offset
  assert at_15_mph.path_angle > 0.0


def test_spatial_slope_stays_quiet_when_action_and_vehicle_already_turn_together():
  controller = LatControlPath()

  command = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.0034,
    measured_curvature=0.01,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.curvature_rate == 0.0


def test_spatial_slope_leads_a_coherent_moving_reversal():
  controller = LatControlPath()

  command = controller.update(
    model=polynomial_model(-0.008, -0.0004),
    desired_curvature=-0.0034,
    measured_curvature=0.004,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.curvature_rate < -0.0001


def test_growing_path_authority_is_bounded_but_release_is_immediate():
  controller = LatControlPath()

  attack = controller.update(polynomial_model(0.05, 0.001), 0.02, 0.0, 7.0, True, False)
  release = controller.update(polynomial_model(0.0), 0.0, 0.0, 7.0, True, False)

  assert 0.0 < attack.path_offset <= 0.5 * 0.006 * 7.0 ** 2
  assert 0.0 < attack.path_angle <= 0.006 * 7.0
  assert 0.0 < attack.curvature_rate <= 0.0002
  assert release.path_offset == 0.0
  assert release.path_angle == 0.0
  assert release.curvature_rate == 0.0


def test_delivered_gentle_model_path_uses_smooth_curvature_alone():
  controller = LatControlPath()

  command = controller.update(
    model=polynomial_model(0.003),
    desired_curvature=0.003,
    measured_curvature=0.003,
    v_ego=15.0,
    active=True,
    driver_override=False,
  )

  assert math.isclose(command.path_offset, 0.0, abs_tol=1e-4)
  assert math.isclose(command.path_angle, 0.0, abs_tol=1e-4)
  assert math.isclose(command.curvature, 0.0002)
  assert command.curvature_rate == 0.0


def test_large_turn_keeps_c2_flushed_until_the_wheel_settles():
  controller = LatControlPath()

  controller.update(polynomial_model(0.015), 0.015, 0.012, 7.0, True, False)
  early_exit = controller.update(polynomial_model(0.004), 0.004, 0.010, 7.0, True, False)
  settled = controller.update(polynomial_model(0.001), 0.001, 0.001, 7.0, True, False)

  assert early_exit.curvature == 0.0
  assert math.isclose(settled.curvature, 0.0002)


def test_inactive_resets_the_previous_command():
  controller = LatControlPath()
  controller.update(polynomial_model(0.015), 0.015, 0.012, 7.0, True, False)

  controller.update(polynomial_model(0.0), 0.0, 0.0, 7.0, False, False)
  reengaged = controller.update(polynomial_model(0.001), 0.001, 0.001, 7.0, True, False)

  assert math.isclose(reengaged.curvature, 0.0002)


def test_missing_model_marks_path_invalid_without_inventing_geometry():
  controller = LatControlPath()

  command = controller.update(None, 0.003, 0.0, 15.0, True, False)

  assert not command.valid
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature_rate == 0.0


def test_nonfinite_model_path_is_invalid():
  controller = LatControlPath()
  model = polynomial_model(0.01)
  model.position.y[3] = float("nan")

  command = controller.update(model, 0.01, 0.0, 7.0, True, False)

  assert not command.valid
