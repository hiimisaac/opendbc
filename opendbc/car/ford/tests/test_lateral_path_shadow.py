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


def test_model_geometry_retains_path_authority_missing_from_action():
  controller = LatControlPath()

  first = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.0034,
    measured_curvature=0.01,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )
  command = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.0034,
    measured_curvature=0.01,
    v_ego=7.0,
    active=True,
    driver_override=False,
  )

  assert command.path_offset > 0.15
  assert command.path_angle > 0.05
  assert math.isclose(first.curvature, 0.0002)


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

  assert command.path_offset == 0.0
  assert math.isclose(command.path_angle, 0.0, abs_tol=1e-9)
  assert math.isclose(command.curvature, 0.0002)
  assert command.curvature_rate == 0.0


def test_large_turn_keeps_c2_flushed_until_path_and_vehicle_settle():
  controller = LatControlPath()

  controller.update(polynomial_model(0.015), 0.015, 0.012, 7.0, True, False)
  early_exit = controller.update(polynomial_model(0.004), 0.004, 0.010, 7.0, True, False)
  for _ in range(9):
    recovering = controller.update(polynomial_model(0.001), 0.001, 0.001, 7.0, True, False)
  released = controller.update(polynomial_model(0.001), 0.001, 0.001, 7.0, True, False)

  assert early_exit.curvature == 0.0
  assert recovering.curvature == 0.0
  assert math.isclose(released.curvature, 0.0002)


def test_inactive_zero_frames_clear_the_large_turn_latch():
  controller = LatControlPath()
  controller.update(polynomial_model(0.015), 0.015, 0.012, 7.0, True, False)

  for _ in range(10):
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
