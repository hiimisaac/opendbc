from types import SimpleNamespace

from opendbc.car.ford.direct_lateral_path import DirectLatControlPath, LateralPathCommand, lmc2_control_utilization


def test_valid_polynomial_reaches_lmc2_without_wheel_feedback_shaping():
  path = SimpleNamespace(
    valid=True,
    pathOffset=0.42,
    pathAngle=-0.08,
    curvature=0.011,
    curvatureRate=-0.0007,
  )

  command = DirectLatControlPath().update(
    path,
    measured_curvature=-0.019,
    v_ego=25.0,
    active=True,
    driver_override=True,
    projected_measured_curvature=0.018,
    desired_angle_curvature=-0.015,
    lat_ctl_limit=2,
  )

  assert command.valid
  assert command.coefficients() == (0.42, -0.08, 0.011, -0.0007)


def test_inactive_control_sends_the_zero_polynomial():
  path = SimpleNamespace(
    valid=True,
    pathOffset=0.42,
    pathAngle=-0.08,
    curvature=0.011,
    curvatureRate=-0.0007,
  )

  command = DirectLatControlPath().update(
    path, measured_curvature=0.0, v_ego=25.0,
    active=False, driver_override=False,
  )

  assert not command.valid
  assert command.coefficients() == (0.0, 0.0, 0.0, 0.0)


def test_direct_polynomial_is_clipped_only_to_the_lmc2_signal_envelope():
  path = SimpleNamespace(
    valid=True,
    pathOffset=10.0,
    pathAngle=-1.0,
    curvature=0.1,
    curvatureRate=-0.01,
  )

  command = DirectLatControlPath().update(
    path, measured_curvature=0.0, v_ego=25.0,
    active=True, driver_override=False,
  )

  assert command.coefficients() == (5.12, -0.5235, 0.02, -0.001023)

  path.curvature = -0.1
  command = DirectLatControlPath().update(
    path, measured_curvature=0.0, v_ego=25.0,
    active=True, driver_override=False,
  )
  assert command.curvature == -0.02


def test_lmc2_control_utilization_tracks_command_and_pscm_limit():
  command = LateralPathCommand(True, 0.0, 0.0, -0.01, 0.0)

  assert lmc2_control_utilization(command, 0) == -0.5
  assert lmc2_control_utilization(command, 1) == -0.8
  assert lmc2_control_utilization(command, 2) == -1.0
  assert lmc2_control_utilization(command, 3) == -0.5
