import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_path import (
  FORD_PATH_C1_CAN_CLIP,
  FORD_PATH_C1_DEADZONE,
  FORD_PATH_C2_CAN_CLIP,
  FORD_PATH_DT,
  FORD_PATH_K_MEAS_TAU,
  FORD_PATH_TRIM_CLIP,
  FORD_PATH_TRIM_KI,
  lateral_path_command,
)


def arc_model(curvature, xs=(0.0, 7.0, 20.0)):
  return SimpleNamespace(
    position=SimpleNamespace(x=list(xs), y=[0.5 * curvature * x * x for x in xs]),
    orientation=SimpleNamespace(z=[curvature * x for x in xs]),
  )


def test_inactive_zeros_command_and_preserves_trim():
  cmd = lateral_path_command(None, 0.01, 0.002, 20.0, 0.005, 0.003, False, False)

  assert cmd.curvature == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0
  assert cmd.k_meas_filt == 0.002
  assert cmd.trim == 0.003


def test_steady_state_arc_idles_c0_c1():
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, k, 20.0, k, 0.0, True, False)

  assert math.isclose(cmd.curvature, k)
  assert math.isclose(cmd.path_angle, 0.0, abs_tol=1e-9)
  assert math.isclose(cmd.path_offset, 0.0, abs_tol=1e-9)


def test_transient_residual_covers_undelivered_curvature():
  # car still going straight, path curves: c1 carries the full lookahead heading
  k = 0.003
  cmd = lateral_path_command(arc_model(k), k, 0.0, 20.0, 0.0, 0.0, True, False)

  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE) * 20.0)
  assert math.isclose(cmd.path_offset, 0.5 * k * 7.0 * 7.0)


def test_c2_rails_and_c1_carries_remainder():
  k = 0.06
  k_held = 0.02  # PSCM already delivering its c2 max
  cmd = lateral_path_command(arc_model(k), k, k_held, 5.0, k_held, 0.0, True, False)

  assert cmd.curvature == FORD_PATH_C2_CAN_CLIP[1]
  # d_look floors at 7m; residual heading past the held arc stays on c1
  assert math.isclose(cmd.path_angle, (k - k_held - FORD_PATH_C1_DEADZONE) * 7.0)


def test_c1_clips_to_can_range():
  cmd = lateral_path_command(None, 0.06, 0.0, 20.0, 0.0, 0.0, True, False)

  assert cmd.path_angle == FORD_PATH_C1_CAN_CLIP[1]


def test_fallback_path_without_model():
  k = 0.002
  cmd = lateral_path_command(None, k, 0.0, 20.0, 0.0, 0.0, True, False)

  assert math.isclose(cmd.path_angle, (k - FORD_PATH_C1_DEADZONE) * 20.0)
  assert math.isclose(cmd.path_offset, 0.5 * k * 7.0 * 7.0)


def test_low_speed_ignores_yaw_curvature():
  # yaw/v curvature is garbage at crawl speed: filter frozen, residual faded out
  cmd = lateral_path_command(None, 0.01, 0.5, 0.5, 0.0, 0.0, True, False)

  assert cmd.k_meas_filt == 0.0
  assert math.isclose(cmd.path_angle, (0.01 - FORD_PATH_C1_DEADZONE) * 7.0)
  assert math.isclose(cmd.path_offset, 0.5 * 0.01 * 7.0 * 7.0)


def test_trim_integrates_persistent_error():
  cmd = lateral_path_command(arc_model(0.001), 0.001, 0.0, 20.0, 0.0, 0.0, True, False)

  assert math.isclose(cmd.trim, FORD_PATH_TRIM_KI * (0.001 - cmd.k_meas_filt) * FORD_PATH_DT)
  assert math.isclose(cmd.curvature, 0.001)  # trim applies from the next step


def test_trim_frozen_when_pressed_slow_or_turning():
  frozen_cases = [
    dict(desired=0.001, v_ego=20.0, pressed=True),
    dict(desired=0.001, v_ego=3.0, pressed=False),
    dict(desired=0.02, v_ego=20.0, pressed=False),
  ]
  for case in frozen_cases:
    cmd = lateral_path_command(arc_model(case["desired"]), case["desired"], 0.0,
                               case["v_ego"], 0.0, 0.001, True, case["pressed"])
    assert cmd.trim == 0.001, case


def test_trim_frozen_when_c2_railed():
  cmd = lateral_path_command(arc_model(0.06), 0.06, 0.0, 20.0, 0.0, 0.004, True, False)

  assert cmd.curvature == FORD_PATH_C2_CAN_CLIP[1]
  assert cmd.trim == 0.004


def test_trim_clipped():
  cmd = lateral_path_command(None, 0.005, 0.005, 20.0, 0.005, 1.0, True, False)

  assert cmd.trim <= FORD_PATH_TRIM_CLIP


def test_k_meas_filter_tracks_measurement():
  alpha = 1.0 - math.exp(-FORD_PATH_DT / FORD_PATH_K_MEAS_TAU)
  cmd = lateral_path_command(None, 0.0, 0.01, 20.0, 0.0, 0.0, True, False)

  assert math.isclose(cmd.k_meas_filt, alpha * 0.01)


def test_non_finite_inputs_are_safe():
  cmd = lateral_path_command(None, float("nan"), float("inf"), 20.0, float("nan"), float("nan"), True, False)

  assert math.isfinite(cmd.curvature)
  assert math.isfinite(cmd.path_angle)
  assert math.isfinite(cmd.path_offset)
