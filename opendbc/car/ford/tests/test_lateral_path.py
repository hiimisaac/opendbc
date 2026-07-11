import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_path import (
  FORD_PATH_C0_UNDERTRACK_ERROR_LIMIT,
  FORD_PATH_C1_CAN_CLIP,
  FORD_PATH_C1_CRUISE_DEADZONE,
  FORD_PATH_C1_DEADZONE,
  FORD_PATH_DT,
  FORD_PATH_K_MEAS_TAU,
  driver_steering_opposes_command,
  lateral_path_command,
)


def arc_model(curvature, xs=(0.0, 7.0, 20.0)):
  return SimpleNamespace(
    position=SimpleNamespace(x=list(xs), y=[0.5 * curvature * x * x for x in xs]),
    orientation=SimpleNamespace(z=[curvature * x for x in xs]),
  )


def test_inactive_zeros_command():
  cmd = lateral_path_command(None, 0.01, 0.002, 20.0, 0.005, False, False)

  assert cmd.curvature == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0
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


def test_driver_override_zeros_entire_command():
  # full relent: the carcontroller drops to mode 0, so nothing may be commanded
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, 0.0, 20.0, 0.0, True, True)

  assert cmd.curvature == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0


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
