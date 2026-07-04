import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_bal import (
  FORD_C2_MEMORY_STACK_GAIN,
  FORD_C2_CANCEL_GAIN,
  FORD_MANEUVER_CURV_THRESH,
  FORD_MANEUVER_ENTER_SCORE,
  blend_lateral_commands,
  c2_authority_step,
  c2_memory_step,
  cancel_c2_from_memory,
  ford_lateral_step,
  lightweight_path_from_curvature,
  lightweight_path_from_model,
  maneuver_score,
  maneuver_score_step,
)


def test_lightweight_path_uses_short_c0_lookahead_at_speed():
  curvature = 0.002
  path_offset, path_angle = lightweight_path_from_curvature(curvature, 20.0, 0.0, True)

  assert math.isclose(path_angle, curvature * 20.0)
  assert math.isclose(path_offset, 0.5 * curvature * 6.0 * 6.0)


def test_lightweight_path_uses_same_low_speed_lookahead_for_c0_and_c1():
  curvature = 0.01
  d_look = 7.0
  path_offset, path_angle = lightweight_path_from_curvature(curvature, 5.0, 0.0, True)

  assert math.isclose(path_angle, curvature * d_look)
  assert math.isclose(path_offset, 0.5 * curvature * d_look * d_look)


def test_lightweight_path_rate_limits_large_c1_reversal():
  _, path_angle = lightweight_path_from_curvature(-0.02, 20.0, 0.497, True)

  assert math.isclose(path_angle, -0.003)


def test_lightweight_path_uses_model_c0_and_c1():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0], y=[0.0, 1.0, 2.0]),
    orientation=SimpleNamespace(z=[0.0, 0.1, 0.2]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, 0.002, 0.002, 20.0, 0.0, True)

  assert math.isclose(path_angle, 0.2)
  model_angle_offset = 0.5 * 0.2 * 6.0 * 6.0 / 20.0
  residual_gain = 0.35 + (20.0 - 15.0) * (0.20 - 0.35) / (30.0 - 15.0)
  assert math.isclose(path_offset, model_angle_offset + (0.6 - model_angle_offset) * residual_gain)


def test_lightweight_path_adds_curvature_error_feedback_to_c1_only():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0], y=[0.0, 0.0, 0.0]),
    orientation=SimpleNamespace(z=[0.0, 0.0, 0.0]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, 0.004, 0.0, 20.0, 0.0, True)

  assert math.isclose(path_angle, 0.004 * (5.0 + (20.0 - 15.0) * (6.0 - 5.0) / (30.0 - 15.0)))
  assert math.isclose(path_offset, 0.0)


def test_lightweight_path_damps_model_c0_residual_more_at_speed():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0, 30.0], y=[0.0, 1.0, 2.0, 3.0]),
    orientation=SimpleNamespace(z=[0.0, 0.1, 0.2, 0.3]),
  )

  low_speed_offset, _ = lightweight_path_from_model(model, 0.002, 0.002, 5.0, 0.0, True)
  high_speed_offset, _ = lightweight_path_from_model(model, 0.002, 0.002, 30.0, 0.0, True)

  assert low_speed_offset > high_speed_offset


def test_lightweight_path_falls_back_to_curvature_without_model():
  assert lightweight_path_from_model(None, 0.002, 0.0, 20.0, 0.0, True) == \
         lightweight_path_from_curvature(0.002, 20.0, 0.0, True)


def test_lightweight_path_from_curvature_subtracts_c2_memory():
  path_offset, path_angle = lightweight_path_from_curvature(0.002, 20.0, 0.0, True, c2_memory=0.002)

  assert math.isclose(path_angle, 0.0)
  assert math.isclose(path_offset, 0.0)


def test_lightweight_path_from_model_subtracts_c2_arc():
  curvature = 0.002
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 6.0, 20.0], y=[0.0, 0.5 * curvature * 6.0 ** 2, 0.5 * curvature * 20.0 ** 2]),
    orientation=SimpleNamespace(z=[0.0, curvature * 6.0, curvature * 20.0]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, curvature, curvature, 20.0, 0.0, True, c2_memory=curvature)

  assert math.isclose(path_angle, 0.0)
  assert math.isclose(path_offset, 0.0)


def test_c2_memory_step_charges_toward_desired_curvature():
  step = c2_memory_step(0.004, 0.002, 0.0, True)

  assert math.isclose(step.command, 0.004)
  assert math.isclose(step.memory, 0.004 * FORD_C2_MEMORY_STACK_GAIN)


def test_c2_memory_step_stops_charging_when_memory_reaches_target():
  step = c2_memory_step(0.004, 0.002, 0.0041, True)

  assert math.isclose(step.command, 0.0)
  assert 0.0 < step.memory < 0.0041


def test_c2_memory_step_stops_charging_when_actual_curvature_is_holding_past_target():
  step = c2_memory_step(0.002, 0.004, 0.0, True)

  assert math.isclose(step.command, 0.0)
  assert math.isclose(step.memory, 0.0)


def test_c2_memory_step_does_not_reverse_pump_stale_memory():
  step = c2_memory_step(-0.002, 0.0, 0.002, True)

  assert math.isclose(step.command, 0.0)
  assert step.memory > 0.0


def test_c2_memory_step_resets_inactive():
  step = c2_memory_step(0.004, 0.004, 0.004, False)

  assert math.isclose(step.command, 0.0)
  assert math.isclose(step.memory, 0.0)


def test_maneuver_score_uses_magnitude_and_rate_only():
  assert math.isclose(maneuver_score(FORD_MANEUVER_CURV_THRESH, 0.012), 1.0)
  assert math.isclose(maneuver_score(0.0, 0.012), 1.0)
  assert maneuver_score(0.002, 0.002) < 0.2
  assert maneuver_score(0.0, 0.0) < 0.2


def test_maneuver_score_step_filters_jitter():
  assert maneuver_score_step(0.012, 0.0, 0.0) < maneuver_score(0.012, 0.0)


def test_c2_authority_stays_high_until_enter_threshold():
  step = c2_authority_step(FORD_MANEUVER_ENTER_SCORE - 0.05, 1.0, True, False)

  assert not step.maneuver_active
  assert step.authority > 0.99


def test_c2_authority_drops_faster_than_it_recovers():
  drop_step = c2_authority_step(1.0, 1.0, True, True)
  recover_step = c2_authority_step(0.0, drop_step.authority, True, False)

  assert drop_step.authority < 1.0
  assert recover_step.authority > drop_step.authority
  assert recover_step.authority <= 1.0


def test_c2_authority_resets_inactive():
  step = c2_authority_step(1.0, 0.2, False, True)

  assert math.isclose(step.authority, 1.0)
  assert math.isclose(step.maneuver_score, 0.0)
  assert not step.maneuver_active


def test_cancel_c2_from_memory_is_disabled_by_default():
  assert math.isclose(cancel_c2_from_memory(0.004), 0.0)
  assert math.isclose(FORD_C2_CANCEL_GAIN, 0.0)


def test_blend_lateral_commands_scales_c2_only():
  apply_curvature, path_offset, path_angle = blend_lateral_commands(0.004, 0.1, 0.2, 0.5)

  assert math.isclose(apply_curvature, 0.002)
  assert math.isclose(path_offset, 0.1)
  assert math.isclose(path_angle, 0.2)


def test_blend_lateral_commands_matches_normal_at_full_authority():
  apply_curvature, path_offset, path_angle = blend_lateral_commands(0.004, 0.1, 0.2, 1.0)

  assert math.isclose(apply_curvature, 0.004)
  assert math.isclose(path_offset, 0.1)
  assert math.isclose(path_angle, 0.2)


def test_ford_lateral_step_suppresses_c2_during_large_maneuver():
  step = ford_lateral_step(
    None, 0.020, 0.020, 0.0, 20.0, 0.0, 0.004, 0.0, 1.0, True, True,
  )

  assert step.maneuver_active
  assert step.c2_authority < 0.5
  assert abs(step.apply_curvature) < abs(c2_memory_step(0.020, 0.0, 0.004, True).command)


def test_ford_lateral_step_matches_base_branch_during_small_request():
  step = ford_lateral_step(
    None, 0.002, 0.002, 0.002, 20.0, 0.0, 0.0, 1.0, 0.0, False, True,
  )
  base_c2 = c2_memory_step(0.002, 0.002, 0.0, True).command
  base_path = lightweight_path_from_curvature(0.002, 20.0, 0.0, True, step.c2_memory)

  assert not step.maneuver_active
  assert math.isclose(step.c2_authority, 1.0)
  assert math.isclose(step.apply_curvature, base_c2)
  assert math.isclose(step.path_offset, base_path[0])
  assert math.isclose(step.path_angle, base_path[1])
