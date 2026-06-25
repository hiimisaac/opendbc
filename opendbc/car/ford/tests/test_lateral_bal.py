import math
from types import SimpleNamespace

from opendbc.car.ford.lateral_bal import lightweight_path_from_curvature, lightweight_path_from_model


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


def test_lightweight_path_keeps_model_c0_and_c1_when_curvature_matches():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0], y=[0.0, 1.0, 2.0]),
    orientation=SimpleNamespace(z=[0.0, 0.1, 0.2]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, 0.01, 0.002, 20.0, 0.0, True)

  assert math.isclose(path_angle, 0.2)
  assert math.isclose(path_offset, 0.6)


def test_lightweight_path_folds_desired_curvature_into_path():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0], y=[0.0, 0.0, 0.0]),
    orientation=SimpleNamespace(z=[0.0, 0.0, 0.0]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, 0.004, 0.004, 20.0, 0.0, True)

  assert math.isclose(path_angle, 0.004 * 20.0)
  assert math.isclose(path_offset, 0.5 * 0.004 * 6.0 * 6.0)


def test_lightweight_path_uses_windowed_model_curvature_for_c1_lead():
  desired_curvature = 0.001
  future_curvature = 0.004
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 14.0], y=[0.0, 0.5 * desired_curvature * 10.0 ** 2, 0.5 * future_curvature * 14.0 ** 2]),
    orientation=SimpleNamespace(z=[0.0, desired_curvature * 10.0, future_curvature * 14.0]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, desired_curvature, 0.0, 10.0, 0.0, True)

  assert math.isclose(path_offset, 0.5 * desired_curvature * 10.0 ** 2)
  assert math.isclose(path_angle, (desired_curvature + (0.25 * (future_curvature - desired_curvature))) * 10.0)


def test_lightweight_path_does_not_lead_against_desired_curvature():
  desired_curvature = -0.001
  future_curvature = 0.004
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 14.0], y=[0.0, 0.5 * desired_curvature * 10.0 ** 2, 0.5 * future_curvature * 14.0 ** 2]),
    orientation=SimpleNamespace(z=[0.0, desired_curvature * 10.0, future_curvature * 14.0]),
  )

  path_offset, path_angle = lightweight_path_from_model(model, desired_curvature, 0.0, 10.0, 0.0, True)

  assert math.isclose(path_offset, 0.5 * desired_curvature * 10.0 ** 2)
  assert math.isclose(path_angle, desired_curvature * 10.0)


def test_lightweight_path_falls_back_to_curvature_without_model():
  assert lightweight_path_from_model(None, 0.002, 0.0, 20.0, 0.0, True) == \
         lightweight_path_from_curvature(0.002, 20.0, 0.0, True)
