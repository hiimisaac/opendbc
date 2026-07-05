import math
from types import SimpleNamespace

from opendbc.car.ford.lmc2_path import (
  c2_memory_decay_step,
  c2_memory_step,
  path_from_model,
  should_use_path_fallback,
)


def test_small_steady_curvature_stays_on_c2_only():
  assert not should_use_path_fallback(0.002, 0.0, True)

  step = c2_memory_step(0.002, 0.0, True)

  assert math.isclose(step.command, 0.002)
  assert step.memory > 0.0


def test_c2_memory_tracks_command_without_stacking():
  c2_step = c2_memory_step(0.0, 0.0, True)
  for _ in range(300):
    c2_step = c2_memory_step(0.002, c2_step.memory, True)

  assert math.isclose(c2_step.command, 0.002)
  assert 0.0019 < c2_step.memory < 0.0021


def test_large_curvature_uses_path_with_c2_zero():
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 10.0, 20.0], y=[0.0, 0.2, 0.4]),
    orientation=SimpleNamespace(z=[0.0, 0.02, 0.04]),
  )

  assert should_use_path_fallback(0.0041, 0.0, True)
  c2_step = c2_memory_decay_step(0.0, True)
  path_offset, path_angle = path_from_model(model, 0.0041, 20.0, 0.0, True, c2_step.memory)

  assert math.isclose(c2_step.command, 0.0)
  assert path_offset > 0.0
  assert path_angle > 0.0


def test_rapid_curvature_change_uses_path_with_c2_zero():
  assert should_use_path_fallback(0.001, 0.021, True)
  assert math.isclose(c2_memory_decay_step(0.002, True).command, 0.0)


def test_stale_c2_memory_does_not_force_path_fallback():
  c2_step = c2_memory_decay_step(0.003, True)

  assert not should_use_path_fallback(0.001, 0.0, True)
  assert not should_use_path_fallback(-0.002, 0.0, True)
  assert c2_step.memory > 0.0


def test_path_fallback_resets_inactive():
  assert not should_use_path_fallback(0.01, 1.0, False)
  assert c2_memory_decay_step(0.01, False) == c2_memory_step(0.01, 0.01, False)
