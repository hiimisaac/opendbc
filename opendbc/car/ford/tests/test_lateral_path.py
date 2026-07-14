import math
from types import SimpleNamespace

from opendbc.car.ford.carcontroller import lmc2_mode, lmc2_precision
from opendbc.car.ford.fordcan import lmc2_curvature_rate_for_can
from opendbc.car.ford.lateral_path import (
  FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT,
  FORD_PATH_C1_CAN_CLIP,
  FORD_PATH_C1_CRUISE_DEADZONE,
  FORD_PATH_C1_DEADZONE,
  FORD_PATH_DT,
  FORD_PATH_K_MEAS_TAU,
  driver_steering_opposes_command,
  lateral_path_command,
  model_curvature_rate,
)


def arc_model(curvature, xs=(0.0, 7.0, 20.0)):
  return SimpleNamespace(
    position=SimpleNamespace(x=list(xs), y=[0.5 * curvature * x * x for x in xs]),
    orientation=SimpleNamespace(z=[curvature * x for x in xs]),
  )


def changing_curvature_model(near_curvature, preview_curvature):
  xs = (0.0, 7.0, 14.0, 20.0)
  curvatures = (0.0, near_curvature, preview_curvature, preview_curvature)
  return SimpleNamespace(
    position=SimpleNamespace(x=list(xs), y=[0.5 * curvature * x * x for x, curvature in zip(xs, curvatures, strict=True)]),
    orientation=SimpleNamespace(z=[curvature * x for x, curvature in zip(xs, curvatures, strict=True)]),
  )


def spatial_curvature_rate_model(curvature, curvature_rate, distances=(0.0, 3.5, 7.0)):
  return SimpleNamespace(
    position=SimpleNamespace(x=list(distances), y=[0.0] * len(distances)),
    orientation=SimpleNamespace(z=[curvature * s + 0.5 * curvature_rate * s * s for s in distances]),
  )


def test_model_curvature_rate_recovers_full_spatial_slope():
  for curvature_rate in (-0.002, -0.0004, 0.0004, 0.002):
    model = spatial_curvature_rate_model(0.01, curvature_rate)
    assert math.isclose(model_curvature_rate(model, 7.0), curvature_rate, abs_tol=1e-12)


def test_model_curvature_rate_uses_distance_along_path():
  # Points are 5 m apart spatially despite advancing only 3 m longitudinally.
  distances = (0.0, 5.0, 10.0)
  curvature_rate = 0.0006
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 3.0, 6.0], y=[0.0, 4.0, 8.0]),
    orientation=SimpleNamespace(z=[0.01 * s + 0.5 * curvature_rate * s * s for s in distances]),
  )

  assert math.isclose(model_curvature_rate(model, 10.0), curvature_rate, abs_tol=1e-12)


def test_model_curvature_rate_rejects_missing_or_degenerate_paths():
  degenerate = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 0.0], y=[0.0, 0.0]),
    orientation=SimpleNamespace(z=[0.0, 0.1]),
  )

  assert model_curvature_rate(None, 7.0) == 0.0
  assert model_curvature_rate(degenerate, 7.0) == 0.0


def test_lmc2_curvature_rate_uses_full_wire_range_without_wrapping():
  assert lmc2_curvature_rate_for_can(0.0008) == 0.0008
  assert lmc2_curvature_rate_for_can(-0.0008) == -0.0008
  assert lmc2_curvature_rate_for_can(0.002) == 0.001023
  assert lmc2_curvature_rate_for_can(-0.002) == -0.001024
  assert lmc2_curvature_rate_for_can(float("nan")) == 0.0


def test_inactive_zeros_command():
  cmd = lateral_path_command(None, 0.01, 0.002, 20.0, 0.005, False, False)

  assert cmd.curvature == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0
  assert cmd.curvature_rate == 0.0
  assert cmd.k_meas_filt == 0.002
  assert cmd.c0_undertrack_correction == 0.0


def test_steady_state_arc_idles_c0_c1():
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  assert math.isclose(cmd.curvature, k)
  assert math.isclose(cmd.path_angle, 0.0, abs_tol=1e-9)
  assert math.isclose(cmd.path_offset, 0.0, abs_tol=1e-9)


def test_transient_residual_covers_undelivered_curvature():
  # car still going straight, path curves: c1 carries the full lookahead heading
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, 0.0, 20.0, 0.0, True, False)

  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE - FORD_PATH_C1_CRUISE_DEADZONE) * 20.0)
  assert cmd.path_offset == 0.0  # c0 is maneuver-only


def test_residual_does_not_accelerate_unwind_when_car_exceeds_path_curvature():
  k = 0.004
  cmd = lateral_path_command(arc_model(k), k, 0.01, 20.0, 0.01, True, False)

  assert cmd.path_angle == 0.0
  assert math.isclose(cmd.path_offset, 0.0, abs_tol=1e-9)


def test_residual_does_not_double_command_during_curvature_reversal():
  k = 0.004
  cmd = lateral_path_command(arc_model(k), k, -0.01, 20.0, -0.01, True, False)

  expected_angle = (k - FORD_PATH_C1_DEADZONE - FORD_PATH_C1_CRUISE_DEADZONE) * 20.0
  expected_offset = 0.5 * k * 7.0 * 7.0 * ((k - 0.003) / (0.006 - 0.003))
  assert math.isclose(cmd.path_angle, expected_angle)
  assert math.isclose(cmd.path_offset, expected_offset)


def test_c2_fades_out_of_maneuvers_and_c1_carries():
  k = 0.06
  k_held = 0.02
  cmd = lateral_path_command(arc_model(k), k, k_held, 5.0, k_held, True, False)

  assert cmd.curvature == 0.0  # maneuver curvature: c2 confined to cruise duty
  # c2 carries nothing, so c1 holds the full arc heading (no delivered-curvature subtraction)
  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE) * 7.0)
  expected_c0_curvature = k + FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT
  assert math.isclose(cmd.path_offset, 0.5 * expected_c0_curvature * 7.0 * 7.0)


def test_c1_holds_arc_mid_maneuver():
  # car already at the turn's curvature with c2 faded out: c1 must NOT collapse,
  # or the polynomial reads "straight" and the turn unwinds early
  k = 0.02
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  assert cmd.curvature == 0.0
  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE) * 20.0)


def test_mid_fade_splits_arc_between_c2_and_c1():
  k = 0.009  # fade midpoint: c2 carries half, c1 sustains the other half
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  assert math.isclose(cmd.curvature, k * 0.5)
  assert math.isclose(cmd.path_angle, (k * 0.5 - FORD_PATH_C1_DEADZONE - 0.5 * FORD_PATH_C1_CRUISE_DEADZONE) * 20.0)


def test_c2_full_strength_on_gentle_curvature():
  k = 0.002
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  assert math.isclose(cmd.curvature, k)


def test_c2_fade_is_continuous():
  k = 0.009  # midpoint of the fade band
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  assert math.isclose(cmd.curvature, k * 0.5)


def test_c1_clips_to_can_range():
  cmd = lateral_path_command(None, 0.06, 0.0, 20.0, 0.0, True, False)

  assert cmd.path_angle == FORD_PATH_C1_CAN_CLIP[1]


def test_fallback_path_without_model():
  k = 0.002
  cmd = lateral_path_command(None, k, 0.0, 20.0, 0.0, True, False)

  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE - FORD_PATH_C1_CRUISE_DEADZONE) * 20.0)
  assert cmd.path_offset == 0.0  # c0 is maneuver-only


def test_low_speed_ignores_yaw_curvature():
  # yaw/v curvature is garbage at crawl speed: filter frozen, residual faded out
  cmd = lateral_path_command(None, 0.02, 0.5, 0.5, 0.0, True, False)

  assert cmd.k_meas_filt == 0.0
  assert math.isclose(cmd.path_angle, (0.02 - FORD_PATH_C1_DEADZONE) * 7.0)
  assert math.isclose(cmd.path_offset, 0.5 * 0.02 * 7.0 * 7.0)


def test_yaw_bias_does_not_create_c2_lane_following_bias():
  # The Lightning's yaw-derived curvature has a persistent positive bias. C2
  # must follow the upstream curvature target, not turn that sensor bias into a
  # real opposite-direction arc through integral feedback.
  k_meas_filt = 0.0
  c2_last = 0.0
  for _ in range(400):
    cmd = lateral_path_command(arc_model(0.0), 0.0, 0.0003, 20.0,
                               k_meas_filt, True, False, c2_last=c2_last)
    k_meas_filt = cmd.k_meas_filt
    c2_last = cmd.curvature

  assert cmd.curvature == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0


def test_driver_override_tracks_delivered_arc_without_c2():
  # Keep LMC2 synchronized to the curve the driver is physically steering so
  # model control can resume in the curve without storing c2/adaptive authority.
  desired = 0.02
  measured = 0.012
  cmd = lateral_path_command(arc_model(desired), desired, measured, 7.0, measured, True, True,
                             path_angle_last=0.4, path_offset_last=1.0)

  assert cmd.curvature == 0.0
  assert math.isclose(cmd.path_angle, measured * 7.0)
  assert math.isclose(cmd.path_offset, 0.5 * measured * 7.0 * 7.0)
  assert cmd.c0_undertrack_correction == 0.0


def test_driver_handoff_blends_back_to_model_path():
  # Logged low-speed hand-back: the manually held arc differed sharply from
  # the model path. Bound both coefficients by one equivalent-curvature step.
  c0_last = 1.339
  c1_last = 0.383
  cmd = lateral_path_command(arc_model(0.0), 0.0, 0.0, 7.0, 0.0, True, False,
                             path_angle_last=c1_last, path_offset_last=c0_last,
                             driver_handoff=True)

  assert math.isclose(cmd.path_angle, c1_last - 0.01 * 7.0)
  assert math.isclose(cmd.path_offset, c0_last - 0.5 * 0.01 * 7.0 * 7.0)
  assert not cmd.handoff_complete


def test_driver_handoff_completes_in_curve():
  path_angle = 0.383
  path_offset = 1.339
  target = lateral_path_command(arc_model(-0.01), -0.01, -0.01, 7.0, -0.01, True, False)

  for _ in range(10):
    cmd = lateral_path_command(arc_model(-0.01), -0.01, -0.01, 7.0, -0.01, True, False,
                               path_angle_last=path_angle, path_offset_last=path_offset,
                               driver_handoff=True)
    path_angle = cmd.path_angle
    path_offset = cmd.path_offset
    if cmd.handoff_complete:
      break

  assert cmd.handoff_complete
  assert math.isclose(cmd.path_angle, target.path_angle)
  assert math.isclose(cmd.path_offset, target.path_offset)


def test_helping_driver_does_not_override_path_command():
  # Logged failed right turn: both request and driver torque were negative.
  driver_override = driver_steering_opposes_command(True, -1.125, -0.013)
  assert not driver_override
  assert not driver_steering_opposes_command(True, 1.125, 0.013)

  cmd = lateral_path_command(arc_model(-0.013), -0.013, 0.0, 4.1, 0.0,
                             True, driver_override, c2_last=0.0)
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0


def test_opposing_driver_overrides_path_command():
  assert driver_steering_opposes_command(True, 1.125, -0.013)
  assert driver_steering_opposes_command(True, -1.125, 0.013)


def test_pressed_driver_overrides_when_there_is_no_path_command():
  assert driver_steering_opposes_command(True, 1.125, 0.0)


def test_zero_torque_during_pressed_release_does_not_interrupt_handoff():
  # steeringPressed is debounced and remains true briefly after torque release.
  assert not driver_steering_opposes_command(True, 0.0, -0.013)


def test_unpressed_driver_never_overrides_path_command():
  assert not driver_steering_opposes_command(False, -1.125, 0.013)


def test_lmc2_stays_comfortably_active_during_driver_override():
  assert lmc2_mode(lat_active=True) == 2
  assert lmc2_precision(cooperative_control=True) == 0
  assert lmc2_mode(lat_active=False) == 0
  assert lmc2_precision(cooperative_control=False) == 1


def test_c2_slew_limits_wire_steps():
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False, c2_last=0.0)

  assert math.isclose(cmd.curvature, 0.0002)  # one slew step toward target


def test_c2_slew_passes_small_tracking():
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False, c2_last=0.0028)

  assert math.isclose(cmd.curvature, k)


def test_k_meas_filter_tracks_measurement():
  alpha = 1.0 - math.exp(-FORD_PATH_DT / FORD_PATH_K_MEAS_TAU)
  cmd = lateral_path_command(None, 0.0, 0.01, 20.0, 0.0, True, False)

  assert math.isclose(cmd.k_meas_filt, alpha * 0.01)


def test_non_finite_inputs_are_safe():
  cmd = lateral_path_command(None, float("nan"), float("inf"), 20.0, float("nan"), True, False)

  assert math.isfinite(cmd.curvature)
  assert math.isfinite(cmd.path_angle)
  assert math.isfinite(cmd.path_offset)
  assert math.isfinite(cmd.curvature_rate)


def test_gentle_lane_following_suppresses_model_spatial_curvature_slope():
  curvature_rate = 0.0008
  cmd = lateral_path_command(spatial_curvature_rate_model(0.003, curvature_rate), 0.003, 0.003, 20.0,
                             0.003, True, False, c2_last=0.003)

  assert cmd.curvature_rate == 0.0


def test_curvature_rate_blends_through_maneuver_entry():
  curvature_rate = 0.0008
  cmd = lateral_path_command(spatial_curvature_rate_model(0.009, curvature_rate), 0.009, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0)

  assert math.isclose(cmd.curvature_rate, 0.5 * curvature_rate, abs_tol=1e-12)


def test_tight_maneuver_carries_full_model_spatial_curvature_slope():
  curvature_rate = 0.0015
  cmd = lateral_path_command(spatial_curvature_rate_model(0.02, curvature_rate), 0.02, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0)

  # Keep the complete model coefficient here; only the physical CAN encoding
  # boundary may saturate it when the message is packed.
  assert math.isclose(cmd.curvature_rate, curvature_rate, abs_tol=1e-12)


def test_c0_full_strength_through_turn_entry():
  # by the time c2 starts fading (0.006), c0 must be at full authority
  k = 0.009
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, True, False)

  expected = 0.5 * k * 7.0 * 7.0 - 0.5 * (k * 0.5) * 7.0 * 7.0  # share=0.5 residual, gain=1.0
  assert math.isclose(cmd.path_offset, expected)


def test_tight_maneuver_path_has_upstream_action_authority():
  # The legacy model path can be weaker than the lag-adjusted upstream action.
  # Once c2 fades out, c0/c1 must still encode at least the requested arc.
  for sign in (-1.0, 1.0):
    desired = sign * 0.02
    cmd = lateral_path_command(arc_model(sign * 0.01), desired, desired, 5.0,
                               desired, True, False, c2_last=0.0)

    assert cmd.curvature == 0.0
    assert math.isclose(cmd.path_offset, 0.5 * desired * 7.0 * 7.0)
    expected_angle = sign * (abs(desired) - FORD_PATH_C1_DEADZONE) * 7.0
    assert math.isclose(cmd.path_angle, expected_angle)


def test_tight_maneuver_keeps_stronger_live_model_geometry_in_c0():
  desired = 0.02
  model_curvature = 0.04
  cmd = lateral_path_command(arc_model(model_curvature), desired, desired, 5.0,
                             desired, True, False, c2_last=0.0)

  model_offset = 0.5 * model_curvature * 7.0 * 7.0
  missing_offset = 0.5 * (model_curvature - desired) * 7.0 * 7.0
  assert math.isclose(cmd.path_offset, model_offset + missing_offset)
  assert math.isclose(cmd.path_angle, (desired - FORD_PATH_C1_DEADZONE) * 7.0)


def test_c1_maneuver_authority_is_bounded_by_model_action():
  # Logged tight turn: model path geometry exceeded the action and drove c1
  # into its wire limit. C0 owns stronger near-path/tight-turn correction.
  desired = -0.025
  near_curvature = -0.03
  preview_curvature = -0.06
  cmd = lateral_path_command(changing_curvature_model(near_curvature, preview_curvature), desired, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0)

  expected_curvature = abs(desired) - FORD_PATH_C1_DEADZONE
  assert math.isclose(cmd.path_angle, -expected_curvature * 7.0)


def test_c1_slew_limits_abrupt_maneuver_entry():
  # A logged model frame changed c1 by 0.136 rad at 7 m in 50 ms. Bound the
  # coefficient by equivalent curvature so the limit scales with lookahead.
  cmd = lateral_path_command(arc_model(0.06), 0.06, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0,
                             path_angle_last=0.0)

  assert math.isclose(cmd.path_angle, 0.006 * 7.0)


def test_c0_slew_limits_abrupt_maneuver_entry():
  cmd = lateral_path_command(arc_model(0.06), 0.06, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0,
                             path_offset_last=0.0)

  assert math.isclose(cmd.path_offset, 0.5 * 0.006 * 7.0 * 7.0)


def test_full_c2_lane_following_only_shapes_maneuver_c0():
  desired = 0.005
  model_curvature = 0.02
  cmd = lateral_path_command(arc_model(model_curvature), desired, 0.0, 7.0,
                             0.0, True, False, c2_last=desired,
                             path_angle_last=0.0, path_offset_last=0.0)

  expected_c1 = (model_curvature - FORD_PATH_C1_DEADZONE - FORD_PATH_C1_CRUISE_DEADZONE) * 7.0
  assert math.isclose(cmd.path_angle, expected_c1)
  assert math.isclose(cmd.path_offset, 0.5 * 0.006 * 7.0 * 7.0)


def test_maneuver_slew_does_not_delay_unwind_or_reversal():
  cmd = lateral_path_command(arc_model(0.0), 0.0, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0,
                             path_angle_last=0.4, path_offset_last=1.0)

  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0

  reverse = lateral_path_command(arc_model(-0.02), -0.02, 0.0, 7.0,
                                 0.0, True, False, c2_last=0.0,
                                 path_angle_last=0.4, path_offset_last=1.0)

  assert reverse.path_angle < 0.0
  assert reverse.path_offset < 0.0


def test_c1_uses_near_constant_curvature():
  curvature = 0.02
  cmd = lateral_path_command(arc_model(curvature, xs=(0.0, 7.0, 14.0, 20.0)), curvature, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0)

  assert math.isclose(cmd.path_angle, (curvature - FORD_PATH_C1_DEADZONE) * 7.0)


def test_c1_ignores_opposite_far_field_curvature():
  desired = 0.02
  near_curvature = 0.02
  preview_curvature = -0.04
  cmd = lateral_path_command(changing_curvature_model(near_curvature, preview_curvature), desired, 0.0, 7.0,
                             0.0, True, False, c2_last=0.0)

  assert math.isclose(cmd.path_angle, (near_curvature - FORD_PATH_C1_DEADZONE) * 7.0)


def test_far_field_geometry_does_not_amplify_near_path_coefficients():
  desired = 0.02
  near_curvature = desired
  preview_curvature = 0.025
  measured = desired
  cmd = lateral_path_command(changing_curvature_model(near_curvature, preview_curvature), desired, measured, 7.0,
                             measured, True, False, c2_last=0.0)

  assert math.isclose(cmd.path_angle, (near_curvature - FORD_PATH_C1_DEADZONE) * 7.0)
  assert math.isclose(cmd.path_offset, 0.5 * near_curvature * 7.0 * 7.0)


def test_c0_assists_undertracking_of_stronger_model_geometry():
  desired = 0.02
  model_curvature = 0.04
  measured = desired
  cmd = lateral_path_command(arc_model(model_curvature, xs=(0.0, 7.0, 14.0, 20.0)), desired, measured, 7.0,
                             measured, True, False, c2_last=0.0)

  model_offset = 0.5 * model_curvature * 7.0 * 7.0
  missing_offset = 0.5 * (model_curvature - measured) * 7.0 * 7.0
  assert math.isclose(cmd.path_offset, model_offset + missing_offset)


def test_tight_maneuver_c0_assists_undertracking():
  desired = 0.03
  measured = 0.015
  cmd = lateral_path_command(arc_model(desired), desired, measured, 7.0,
                             measured, True, False, c2_last=0.0)

  requested_offset = 0.5 * desired * 7.0 * 7.0
  missing_offset = 0.5 * (desired - measured) * 7.0 * 7.0
  assert math.isclose(cmd.path_offset, requested_offset + missing_offset)


def test_c0_undertracking_assist_is_bounded_to_useful_state():
  desired = 0.03
  requested_offset = 0.5 * desired * 7.0 * 7.0
  cases = [
    (0.04, 7.0),   # already tracking beyond the request
    (0.015, 1.5),  # yaw/v is unreliable at crawl speed
    (0.015, 15.0), # avoid exciting high-speed c0 hunting
  ]
  for measured, v_ego in cases:
    cmd = lateral_path_command(arc_model(desired), desired, measured, v_ego,
                               measured, True, False, c2_last=0.0)
    assert math.isclose(cmd.path_offset, requested_offset), (measured, v_ego)


def test_persistent_logged_undertracking_builds_c0_only_authority():
  # Logged steady alert near 24 mph: model requested ~0.021/m while the
  # Lightning held ~0.015/m. Persistent error may earn more c0, but must not
  # change c1/c2 or apply a universal path gain.
  desired = 0.021
  measured = 0.015
  v_ego = 10.5
  correction = 0.0
  commands = []
  for _ in range(12):
    cmd = lateral_path_command(arc_model(desired), desired, measured, v_ego,
                               measured, True, False, c2_last=0.0,
                               c0_undertrack_correction=correction)
    correction = cmd.c0_undertrack_correction
    commands.append(cmd)

  assert commands[-1].path_offset > commands[0].path_offset
  assert commands[-1].path_angle == commands[0].path_angle
  assert commands[-1].curvature == commands[0].curvature
  assert 0.0 < correction <= 0.01


def test_c0_adaptation_resets_before_reversed_turn():
  correction = 0.006
  cmd = lateral_path_command(arc_model(-0.02), -0.02, 0.015, 7.0,
                             0.015, True, False, c2_last=0.0,
                             c0_undertrack_correction=correction)

  assert cmd.c0_undertrack_correction == 0.0


def test_c0_adaptation_releases_when_tracking_catches_up():
  correction = 0.006
  cmd = lateral_path_command(arc_model(0.02), 0.02, 0.02, 7.0,
                             0.02, True, False, c2_last=0.0,
                             c0_undertrack_correction=correction)

  assert 0.0 <= cmd.c0_undertrack_correction < correction


def test_c0_adaptation_is_disabled_in_gentle_lane_following():
  correction = 0.0
  for _ in range(100):
    cmd = lateral_path_command(arc_model(0.003), 0.003, 0.0, 20.0,
                               0.0, True, False, c2_last=0.003,
                               c0_undertrack_correction=correction)
    correction = cmd.c0_undertrack_correction

  assert correction == 0.0


def test_c0_adaptation_is_bounded_in_persistent_tight_turn():
  correction = 0.0
  for _ in range(200):
    cmd = lateral_path_command(arc_model(0.06), 0.06, 0.0, 7.0,
                               0.0, True, False, c2_last=0.0,
                               c0_undertrack_correction=correction)
    correction = cmd.c0_undertrack_correction

  assert math.isclose(correction, 0.01, rel_tol=1e-6)


def test_c0_adaptation_does_not_wind_up_behind_c0_limit():
  correction = 0.0
  for _ in range(200):
    cmd = lateral_path_command(arc_model(0.2), 0.2, 0.0, 7.0,
                               0.0, True, False, c2_last=0.0,
                               c0_undertrack_correction=correction)
    correction = cmd.c0_undertrack_correction

  assert cmd.path_offset == 4.60
  assert correction == 0.0


def test_c0_adaptation_clears_on_driver_override():
  cmd = lateral_path_command(arc_model(0.02), 0.02, 0.01, 7.0,
                             0.01, True, True, c2_last=0.0,
                             c0_undertrack_correction=0.006)

  assert cmd.c0_undertrack_correction == 0.0
