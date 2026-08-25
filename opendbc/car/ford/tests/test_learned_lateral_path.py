import pytest
from types import SimpleNamespace

from opendbc.car.ford.learned_lateral_path import (
  LearnedLateralPathCommand,
  LearnedLateralPathController,
  lmc2_control_utilization,
  PathPolynomial,
  SteeringAngleProjector,
)


def test_path_polynomial_advances_exactly_in_space():
  path = PathPolynomial(-0.4, -0.05, -0.008, -0.0002)

  advanced = path.advanced(3.0)

  assert advanced.c0 == pytest.approx(-0.4 - 0.05 * 3.0 - 0.5 * 0.008 * 3.0**2 - 0.0002 * 3.0**3 / 6.0)
  assert advanced.c1 == pytest.approx(-0.05 - 0.008 * 3.0 - 0.5 * 0.0002 * 3.0**2)
  assert advanced.c2 == pytest.approx(-0.008 - 0.0002 * 3.0)
  assert advanced.c3 == pytest.approx(-0.0002)


def test_path_polynomial_builds_causal_future_targets_and_wire_commands():
  path = PathPolynomial(-0.4, -0.05, -0.008, -0.0002)

  horizons = path.horizons(speed_mps=10.0, timestep_s=0.05, steps=3)

  assert [sample.c2 for sample in horizons] == pytest.approx([-0.0081, -0.0082, -0.0083])
  assert horizons[0].as_wire_coefficients() == pytest.approx(tuple(-value for value in horizons[0].coefficients()))
  assert path.desired_angles_deg(
    speed_mps=10.0,
    timestep_s=0.05,
    steps=3,
    current_desired_angle_deg=42.0,
    curvature_to_angle_deg=lambda curvature: 1000.0 * curvature,
  ) == pytest.approx([42.1, 42.2, 42.3])


def test_angle_projector_exposes_causal_20hz_rate():
  projector = SteeringAngleProjector()

  projector.update(10.0)
  projected = projector.update(11.0)

  assert projector.rate_deg_s == pytest.approx(20.0)
  assert projected == pytest.approx(18.0)


def test_lmc2_utilization_preserves_ui_meter_without_legacy_controller():
  command = LearnedLateralPathCommand(True, 0.0, 0.0, 0.01, 0.0)

  assert lmc2_control_utilization(command, 0) == pytest.approx(0.5)
  assert lmc2_control_utilization(command, 1) == pytest.approx(0.8)
  assert lmc2_control_utilization(command, 2) == pytest.approx(1.0)


def test_learned_controller_is_bounded_and_inactive_is_zero():
  controller = LearnedLateralPathController()
  path = SimpleNamespace(valid=True, pathOffset=0.59, pathAngle=0.031, curvature=-0.00137, curvatureRate=-0.00102)

  command = controller.update(
    path, desired_angle_deg=5.28, actual_angle_deg=-61.3, steering_rate_deg_s=-24.0,
    speed_mps=9.0, eps_current_a=8.0, projected_curvature=0.0171,
    measured_curvature=0.0171, desired_curvature=-0.00147, lat_ctl_limit=0, active=True,
  )

  assert command.valid
  assert -5.11 <= command.path_offset <= 5.12
  assert -0.5235 <= command.path_angle <= 0.5
  assert -0.02 <= command.curvature <= 0.02
  assert -0.001023 <= command.curvature_rate <= 0.001024
  assert command.path_offset == pytest.approx(-0.21)
  assert command.path_angle == pytest.approx(0.006)
  assert command.curvature == pytest.approx(0.00024)
  assert command.curvature_rate == pytest.approx(-0.00037)
  assert controller.update(
    path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False,
  ).valid is False
