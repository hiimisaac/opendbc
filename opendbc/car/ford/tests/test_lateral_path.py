import math
from types import SimpleNamespace

from opendbc.car.ford.carcontroller import (
  _load_messaging,
  ford_curvature_from_steering_angle,
  lmc2_mode,
  lmc2_precision,
)
from opendbc.car.ford.fordcan import lmc2_curvature_rate_for_can
from opendbc.car.ford.lateral_path import (
  FORD_PATH_C0_FEEDBACK_ERROR_LIMIT,
  FORD_PATH_C1_CAN_CLIP,
  FORD_PATH_C1_CRUISE_DEADZONE,
  FORD_PATH_C1_DEADZONE,
  FORD_PATH_DT,
  FORD_PATH_K_MEAS_TAU,
  SteeringAngleProjector,
  driver_steering_opposes_command,
  lateral_path_command,
  model_curvature_rate,
  model_curvature_rate_consensus,
  projected_tracking_error,
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


def spatial_curvature_rate_model(curvature, curvature_rate,
                                 distances=(0.0, 1.75, 2.5, 3.5, 5.0, 7.0)):
  return SimpleNamespace(
    position=SimpleNamespace(x=list(distances), y=[0.0] * len(distances)),
    orientation=SimpleNamespace(z=[curvature * s + 0.5 * curvature_rate * s * s for s in distances]),
  )


def polynomial_path_model(curvature, curvature_rate,
                          distances=(0.0, 1.75, 2.5, 3.5, 5.0, 7.0)):
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * curvature * s * s + curvature_rate * s * s * s / 6.0 for s in distances],
    ),
    orientation=SimpleNamespace(z=[curvature * s + 0.5 * curvature_rate * s * s for s in distances]),
  )


def multi_horizon_curvature_rate_model(rate_3_5, rate_5, rate_7):
  distances = (0.0, 1.75, 2.5, 3.5, 5.0, 7.0)
  heading_3_5 = rate_3_5 * 3.5 * 3.5 / 4.0
  headings = (0.0, 0.0, 0.0, heading_3_5,
              rate_5 * 5.0 * 5.0 / 4.0,
              2.0 * heading_3_5 + rate_7 * 7.0 * 7.0 / 4.0)
  return SimpleNamespace(
    position=SimpleNamespace(x=list(distances), y=[0.0] * len(distances)),
    orientation=SimpleNamespace(z=list(headings)),
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


def test_model_curvature_rate_consensus_keeps_full_same_direction_median():
  model = multi_horizon_curvature_rate_model(0.0002, 0.0008, 0.002)

  assert math.isclose(model_curvature_rate_consensus(model), 0.0008, abs_tol=1e-12)


def test_model_curvature_rate_consensus_attenuates_horizon_disagreement():
  model = multi_horizon_curvature_rate_model(-0.0002, 0.0008, 0.001)

  # Median slope times |sum| / sum(|slope|): 0.0008 * 0.0016 / 0.002.
  assert math.isclose(model_curvature_rate_consensus(model), 0.00064, abs_tol=1e-12)


def test_model_curvature_rate_consensus_cancels_balanced_disagreement():
  model = multi_horizon_curvature_rate_model(-0.0008, 0.0, 0.0008)

  assert model_curvature_rate_consensus(model) == 0.0


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
  assert math.isclose(cmd.path_offset, 0.5 * k * 7.0 * 7.0)


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


def test_large_turn_c2_does_not_reenter_during_unwind():
  # Reproduce the logged failure: after a large turn makes c2 zero, a falling
  # desired curvature must not let old-direction c2 fade back in.
  c2_last = 0.0
  c2_latched = False
  recovery_frames = 0
  unwind_curvature = 0.0

  for desired in (0.012, 0.009, 0.006, 0.003, 0.0):
    wheel_curvature = 0.012
    cmd = lateral_path_command(
      arc_model(desired), desired, wheel_curvature, 7.0, wheel_curvature, True, False,
      c2_last=c2_last,
      angle_error_curvature=desired - wheel_curvature,
      wheel_curvature=wheel_curvature,
      c2_latched_last=c2_latched,
      c2_recovery_frames_last=recovery_frames,
      unwind_curvature_last=unwind_curvature,
    )
    c2_last = cmd.curvature
    c2_latched = cmd.c2_latched
    recovery_frames = cmd.c2_recovery_frames
    unwind_curvature = cmd.unwind_curvature
    assert cmd.curvature == 0.0

  # At zero target the wheel is still in the old turn, so c0/c1 must carry a
  # bounded opposite-direction return instead of merely dropping to zero.
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0


def test_large_turn_c2_releases_only_after_stable_physical_flush():
  c2_latched = True
  recovery_frames = 0
  for frame in range(10):
    cmd = lateral_path_command(
      arc_model(0.0), 0.0, 0.0, 7.0, 0.0, True, False,
      c2_last=0.0,
      angle_error_curvature=0.0,
      wheel_curvature=0.0,
      c2_latched_last=c2_latched,
      c2_recovery_frames_last=recovery_frames,
    )
    c2_latched = cmd.c2_latched
    recovery_frames = cmd.c2_recovery_frames
    assert c2_latched == (frame < 9)


def test_quick_disengage_preserves_large_turn_c2_flush():
  inactive = lateral_path_command(
    None, 0.0, 0.0, 7.0, 0.0, False, False,
    c2_latched_last=True,
  )
  reengaged = lateral_path_command(
    arc_model(0.003), 0.003, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0,
    c2_latched_last=inactive.c2_latched,
    c2_recovery_frames_last=inactive.c2_recovery_frames,
  )

  assert inactive.c2_latched
  assert reengaged.c2_latched
  assert reengaged.curvature == 0.0


def test_large_turn_unwind_only_corrects_toward_desired_wheel_angle():
  common = dict(c2_last=0.0, c2_latched_last=True, c2_recovery_frames_last=0)

  lagging = lateral_path_command(
    arc_model(0.003), 0.003, 0.01, 7.0, 0.01, True, False,
    angle_error_curvature=-0.007, wheel_curvature=0.01, **common,
  )
  ahead = lateral_path_command(
    arc_model(0.003), 0.003, 0.001, 7.0, 0.001, True, False,
    angle_error_curvature=0.002, wheel_curvature=0.001, **common,
  )
  unlatched = lateral_path_command(
    arc_model(0.003), 0.003, 0.01, 7.0, 0.01, True, False,
    angle_error_curvature=-0.007, wheel_curvature=0.01,
  )

  assert lagging.unwind_curvature < 0.0
  assert ahead.unwind_curvature == 0.0
  assert unlatched.unwind_curvature == 0.0


def test_large_turn_unwind_does_not_fight_same_direction_angle_feedback():
  # Steering angle can show the wheel still wound into the old turn while the
  # upstream path has begun its exit. During this disagreement the physical
  # wheel owns the exit direction.
  cmd = lateral_path_command(
    arc_model(0.003), 0.003, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0,
    angle_error_curvature=-0.009,
    wheel_curvature=0.012,
    projected_wheel_curvature=0.006,
    c2_latched_last=True,
  )

  assert cmd.unwind_curvature < 0.0
  assert cmd.path_offset < 0.0


def test_large_turn_unwind_rejects_only_opposing_model_curvature_rate():
  common = dict(
    desired_curvature=0.013,
    k_meas=0.02,
    v_ego=7.0,
    k_meas_filt=0.02,
    lat_active=True,
    driver_override=False,
    c2_last=0.0,
    angle_error_curvature=-0.007,
    wheel_curvature=0.02,
    c2_latched_last=True,
  )
  opposing = lateral_path_command(spatial_curvature_rate_model(0.013, 0.0008), **common)
  assisting = lateral_path_command(spatial_curvature_rate_model(0.013, -0.0008), **common)

  assert opposing.curvature_rate == 0.0
  assert math.isclose(assisting.curvature_rate, -0.0008, abs_tol=1e-12)


def test_gentle_lane_following_ignores_wheel_angle_feedback():
  baseline = lateral_path_command(arc_model(0.003), 0.003, 0.0, 20.0, 0.0, True, False,
                                  c2_last=0.003)
  with_angle_error = lateral_path_command(
    arc_model(0.003), 0.003, 0.0, 20.0, 0.0, True, False,
    c2_last=0.003, angle_error_curvature=-0.02, wheel_curvature=0.02,
  )

  assert with_angle_error.curvature == baseline.curvature
  assert with_angle_error.curvature_rate == baseline.curvature_rate
  assert with_angle_error.path_angle == baseline.path_angle
  assert with_angle_error.path_offset == baseline.path_offset


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
  # model control can resume in the curve without storing c2/feedback authority.
  desired = 0.02
  measured = 0.012
  cmd = lateral_path_command(arc_model(desired), desired, measured, 7.0, measured, True, True,
                             path_angle_last=0.4, path_offset_last=1.0,
                             projected_wheel_curvature=0.0)

  assert cmd.curvature == 0.0
  assert math.isclose(cmd.path_angle, measured * 7.0)
  assert math.isclose(cmd.path_offset, 0.5 * measured * 7.0 * 7.0)


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
  # Logged handoff: desired wheel motion and driver torque were both negative.
  # Curvature has the opposite sign on Ford, so classify in wheel-angle space.
  driver_override = driver_steering_opposes_command(True, -1.125, -15.0)
  assert not driver_override
  assert not driver_steering_opposes_command(True, 1.125, 15.0)

  cmd = lateral_path_command(arc_model(-0.013), -0.013, 0.0, 4.1, 0.0,
                             True, driver_override, c2_last=0.0)
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0


def test_opposing_driver_overrides_path_command():
  assert driver_steering_opposes_command(True, 1.125, -15.0)
  assert driver_steering_opposes_command(True, -1.125, 15.0)


def test_pressed_driver_overrides_when_there_is_no_path_command():
  assert driver_steering_opposes_command(True, 1.125, 0.0)


def test_zero_torque_during_pressed_release_does_not_interrupt_handoff():
  # steeringPressed is debounced and remains true briefly after torque release.
  assert not driver_steering_opposes_command(True, 0.0, -15.0)


def test_unpressed_driver_never_overrides_path_command():
  assert not driver_steering_opposes_command(False, -1.125, 0.013)


def test_sunnypilot_messaging_namespace_is_preferred():
  sentinel = object()
  imports = []

  def fake_import(name):
    imports.append(name)
    if name == "cereal.messaging":
      return sentinel
    raise ImportError(name)

  assert _load_messaging(fake_import) is sentinel
  assert imports == ["cereal.messaging"]


def test_upstream_messaging_namespace_is_the_fallback():
  sentinel = object()
  imports = []

  def fake_import(name):
    imports.append(name)
    if name == "openpilot.cereal.messaging":
      return sentinel
    raise ImportError(name)

  assert _load_messaging(fake_import) is sentinel
  assert imports == ["cereal.messaging", "openpilot.cereal.messaging"]


def test_missing_messaging_namespaces_disable_model_subscription():
  def fake_import(name):
    raise ImportError(name)

  assert _load_messaging(fake_import) is None


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


def test_c2_release_is_immediate_and_reversal_flushes_old_direction():
  release = lateral_path_command(arc_model(0.001), 0.001, 0.001, 20.0, 0.001,
                                 True, False, c2_last=0.004)
  reversal = lateral_path_command(arc_model(0.003), 0.003, 0.003, 20.0, 0.003,
                                  True, False, c2_last=-0.004)
  reversed_attack = lateral_path_command(arc_model(0.003), 0.003, 0.003, 20.0, 0.003,
                                         True, False, c2_last=reversal.curvature)

  assert math.isclose(release.curvature, 0.001)
  assert reversal.curvature == 0.0
  assert math.isclose(reversed_attack.curvature, 0.0002)


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


def test_coherent_polynomial_preview_starts_moving_turn_before_live_action_gate():
  # Logged moving-turn reversal: the current action is still gentle, but c0,
  # c1, and c3 already agree that a real turn begins in the near field.
  for sign in (-1.0, 1.0):
    cmd = lateral_path_command(
      polynomial_path_model(sign * 0.0025, sign * 0.0015), sign * 0.0025,
      -sign * 0.0015, 7.0, -sign * 0.0015, True, False,
      c2_last=sign * 0.0025, path_angle_last=0.0, path_offset_last=0.0, curvature_rate_last=0.0,
    )

    assert math.isclose(cmd.curvature, sign * 0.0025, abs_tol=1e-12)
    assert math.isclose(cmd.curvature_rate, sign * 0.0002, abs_tol=1e-12)
    assert cmd.path_offset * sign > 0.0


def test_polynomial_preview_requires_motion_and_delivered_curvature_reversal():
  common = dict(
    model=polynomial_path_model(0.0025, 0.0015), desired_curvature=0.0025,
    lat_active=True, driver_override=False, c2_last=0.0025,
    path_angle_last=0.0, path_offset_last=0.0, curvature_rate_last=0.0,
  )
  stopped = lateral_path_command(k_meas=-0.0015, v_ego=0.0, k_meas_filt=-0.0015, **common)
  same_direction = lateral_path_command(k_meas=0.0015, v_ego=7.0, k_meas_filt=0.0015, **common)

  assert stopped.curvature_rate == 0.0
  assert stopped.path_offset == 0.0
  assert same_direction.curvature_rate == 0.0
  assert same_direction.path_offset == 0.0


def test_polynomial_preview_requires_c0_c1_c3_direction_agreement():
  model = polynomial_path_model(0.0025, 0.0015)
  model.position.y = [-y for y in model.position.y]
  cmd = lateral_path_command(
    model, 0.0025, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0025, path_angle_last=0.0, path_offset_last=0.0, curvature_rate_last=0.0,
  )

  assert cmd.curvature_rate == 0.0
  assert cmd.path_offset == 0.0


def test_polynomial_preview_rejects_conflicting_path_heading():
  model = polynomial_path_model(0.0025, 0.0015)
  model.orientation.z = [heading - 0.01 * distance
                         for heading, distance in zip(model.orientation.z, model.position.x, strict=True)]
  cmd = lateral_path_command(
    model, 0.0025, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0025, path_angle_last=0.0, path_offset_last=0.0, curvature_rate_last=0.0,
  )

  assert cmd.curvature_rate == 0.0
  assert cmd.path_offset == 0.0


def test_polynomial_preview_does_not_hold_geometry_during_turn_exit():
  cmd = lateral_path_command(
    polynomial_path_model(0.01, -0.0015), 0.0025, 0.006, 7.0, 0.006, True, False,
    c2_last=0.0025, path_angle_last=0.0, path_offset_last=0.0, curvature_rate_last=0.0,
  )

  assert cmd.curvature_rate == 0.0
  assert cmd.path_offset == 0.0


def test_large_turn_latch_does_not_force_c3_into_gentle_tracking():
  cmd = lateral_path_command(spatial_curvature_rate_model(0.003, 0.0008), 0.003, 0.003, 7.0,
                             0.003, True, False, c2_last=0.0, c2_latched_last=True)

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


def test_curvature_rate_keeps_spatial_consensus_during_action_reversal():
  previous = lateral_path_command(
    spatial_curvature_rate_model(0.02, 0.0008), 0.020, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0, curvature_rate_last=0.0008,
  )
  cmd = lateral_path_command(
    spatial_curvature_rate_model(0.02, 0.0008), 0.019, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0, curvature_rate_last=previous.curvature_rate,
  )

  assert math.isclose(cmd.curvature_rate, 0.0008, abs_tol=1e-12)


def test_curvature_rate_limits_attack_but_reaches_full_spatial_slope():
  curvature_rate = 0.0015
  last = 0.0
  for frame in range(8):
    cmd = lateral_path_command(
      spatial_curvature_rate_model(0.02, curvature_rate), 0.021, 0.0, 7.0, 0.0, True, False,
      c2_last=0.0, curvature_rate_last=last,
    )
    last = cmd.curvature_rate
    if frame == 0:
      assert math.isclose(last, 0.0002)

  assert math.isclose(last, curvature_rate, abs_tol=1e-12)


def test_curvature_rate_reversal_is_not_delayed():
  cmd = lateral_path_command(
    spatial_curvature_rate_model(-0.02, -0.0008), -0.021, 0.0, 7.0, 0.0, True, False,
    c2_last=0.0, curvature_rate_last=0.0008,
  )

  assert math.isclose(cmd.curvature_rate, -0.0008, abs_tol=1e-12)


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
  assert math.isclose(cmd.path_offset, model_offset)
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


def test_c0_feedback_tracks_stronger_model_geometry_than_action():
  desired = 0.02
  model_curvature = 0.04
  measured = desired
  cmd = lateral_path_command(arc_model(model_curvature, xs=(0.0, 7.0, 14.0, 20.0)), desired, measured, 7.0,
                             measured, True, False, c2_last=0.0,
                             wheel_curvature=measured,
                             projected_wheel_curvature=measured)

  model_offset = 0.5 * model_curvature * 7.0 * 7.0
  missing_offset = 0.5 * (model_curvature - measured) * 7.0 * 7.0
  assert math.isclose(cmd.path_offset, model_offset + missing_offset)


def test_yaw_undertracking_does_not_add_feedback_without_steering_error():
  desired = 0.03
  measured = 0.015
  cmd = lateral_path_command(arc_model(desired), desired, measured, 7.0,
                             measured, True, False, c2_last=0.0,
                             wheel_curvature=desired,
                             projected_wheel_curvature=desired)

  requested_offset = 0.5 * desired * 7.0 * 7.0
  assert math.isclose(cmd.path_offset, requested_offset)


def test_angle_feedback_is_maneuver_only_and_never_countersteers():
  desired = 0.03
  requested_offset = 0.5 * desired * 7.0 * 7.0
  countersteer = lateral_path_command(
    arc_model(desired), desired, 0.0, 7.0, 0.0, True, False, c2_last=0.0,
    wheel_curvature=0.05, projected_wheel_curvature=0.05,
  )
  gentle = lateral_path_command(
    arc_model(0.003), 0.003, 0.0, 20.0, 0.0, True, False, c2_last=0.003,
    wheel_curvature=0.0, projected_wheel_curvature=0.0,
  )

  assert math.isclose(countersteer.path_offset, requested_offset)
  assert gentle.path_offset == 0.0


def test_angle_feedback_is_bounded_and_c0_only():
  desired = 0.03
  measured = 0.0
  base = lateral_path_command(arc_model(desired), desired, measured, 7.0,
                              measured, True, False, c2_last=0.0,
                              wheel_curvature=desired, projected_wheel_curvature=desired)
  feedback = lateral_path_command(
    arc_model(desired), desired, measured, 7.0, measured, True, False, c2_last=0.0,
    wheel_curvature=0.0, projected_wheel_curvature=0.0,
  )

  expected_added_offset = 0.5 * FORD_PATH_C0_FEEDBACK_ERROR_LIMIT * 7.0 * 7.0
  assert math.isclose(feedback.path_offset, base.path_offset + expected_added_offset)
  assert feedback.path_angle == base.path_angle
  assert feedback.curvature == base.curvature


def test_angle_feedback_has_no_persistent_state():
  desired = 0.03
  measured = 0.0
  common = dict(model=arc_model(desired), desired_curvature=desired, k_meas=measured,
                v_ego=7.0, k_meas_filt=measured, lat_active=True,
                driver_override=False, c2_last=0.0)

  with_feedback = lateral_path_command(**common, wheel_curvature=0.0, projected_wheel_curvature=0.0)
  released = lateral_path_command(**common, wheel_curvature=desired, projected_wheel_curvature=desired)

  assert with_feedback.path_offset > released.path_offset
  assert released.path_offset == lateral_path_command(
    **common, wheel_curvature=desired, projected_wheel_curvature=desired,
  ).path_offset


def test_predicted_steering_error_reduces_c0_feedback_as_wheel_closes():
  desired = 0.03
  measured = desired
  common = dict(model=arc_model(desired), desired_curvature=desired, k_meas=measured,
                v_ego=7.0, k_meas_filt=measured, lat_active=True,
                driver_override=False, c2_last=0.0)

  base = lateral_path_command(**common, wheel_curvature=desired, projected_wheel_curvature=desired)
  closing = lateral_path_command(**common, wheel_curvature=0.0, projected_wheel_curvature=0.025)
  lagging = lateral_path_command(**common, wheel_curvature=0.0, projected_wheel_curvature=0.015)

  assert base.path_offset < closing.path_offset < lagging.path_offset


def test_projection_only_discounts_curvature_error_closing_model_target():
  assert math.isclose(projected_tracking_error(0.04, 0.02, 0.03), 0.01)
  assert projected_tracking_error(0.04, 0.02, 0.05) == 0.0
  assert math.isclose(projected_tracking_error(0.04, 0.02, 0.01), 0.02)
  assert math.isclose(projected_tracking_error(-0.04, -0.02, -0.03), -0.01)


def test_steering_angle_projector_uses_exact_20hz_tenth_second_window():
  projector = SteeringAngleProjector()

  assert projector.update(50.0) == 50.0
  assert projector.update(55.0) == 65.0
  assert projector.update(60.0) == 70.0


def test_ford_steering_angle_conversion_has_curvature_sign():
  vm = SimpleNamespace(calc_curvature=lambda angle, _speed, _roll: angle)

  assert math.isclose(ford_curvature_from_steering_angle(vm, 90.0, 7.0), -math.pi / 2.0)
  assert math.isclose(ford_curvature_from_steering_angle(vm, -90.0, 7.0), math.pi / 2.0)
