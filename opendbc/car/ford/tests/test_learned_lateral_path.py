import pytest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from opendbc.car.ford.learned_lateral_path import (
  AdaptiveLateralTrim,
  LearnedLateralPathCommand,
  LearnedLateralPathController,
  lmc2_control_utilization,
  PathPolynomial,
  SteeringAngleProjector,
)


def test_adaptive_trim_disabled_preserves_nominal_command_exactly():
  trim = AdaptiveLateralTrim(enabled=False)
  nominal = LearnedLateralPathCommand(True, 0.42, -0.031, 0.004, -0.0002)

  command = trim.update(
    nominal, desired_curvature=0.02, measured_curvature=0.0,
    active=True, driver_input=False, lat_ctl_limit=0,
  )

  assert command == nominal


def test_adaptive_trim_learns_coherent_fast_authority_without_changing_c2():
  trim = AdaptiveLateralTrim(enabled=True)
  nominal = LearnedLateralPathCommand(True, 0.4, 0.04, 0.003, 0.0003)

  command = nominal
  for _ in range(200):
    command = trim.update(
      nominal, desired_curvature=0.02, measured_curvature=0.0,
      active=True, driver_input=False, lat_ctl_limit=0,
    )

  assert 1.0 < command.path_offset / nominal.path_offset <= 1.12
  assert command.path_angle / nominal.path_angle == pytest.approx(command.path_offset / nominal.path_offset)
  assert command.curvature_rate / nominal.curvature_rate == pytest.approx(command.path_offset / nominal.path_offset)
  assert command.curvature == nominal.curvature


def test_adaptive_trim_does_not_modify_straight_commands_after_learning():
  trim = AdaptiveLateralTrim(enabled=True)
  turn = LearnedLateralPathCommand(True, 0.4, 0.04, 0.003, 0.0003)
  for _ in range(200):
    trim.update(turn, 0.02, 0.0, True, False, 0)
  assert trim.state.gain > 0.0

  straight = LearnedLateralPathCommand(True, 0.02, 0.001, 0.0002, 0.00001)
  command = trim.update(straight, 0.001, 0.001, True, False, 0)

  assert command == straight
  assert not trim.state.adapting


@pytest.mark.parametrize("driver_input,lat_ctl_limit", [(True, 0), (False, 1), (False, 2)])
def test_adaptive_trim_freezes_during_override_and_pscm_limits(driver_input, lat_ctl_limit):
  trim = AdaptiveLateralTrim(enabled=True)
  nominal = LearnedLateralPathCommand(True, 0.4, 0.04, 0.003, 0.0003)
  for _ in range(200):
    trim.update(nominal, 0.02, 0.0, True, False, 0)
  learned_gain = trim.state.gain

  command = trim.update(nominal, 0.02, 0.0, True, driver_input, lat_ctl_limit)

  assert not trim.state.adapting
  assert trim.state.gain == pytest.approx(learned_gain)
  assert command.path_offset == pytest.approx(nominal.path_offset * (1.0 + learned_gain))


def test_adaptive_trim_uses_projected_arrival_to_taper_before_overshoot():
  trim = AdaptiveLateralTrim(enabled=True)
  nominal = LearnedLateralPathCommand(True, 0.4, 0.04, 0.003, 0.0003)

  command = nominal
  for _ in range(200):
    command = trim.update(
      nominal, desired_curvature=0.02, measured_curvature=0.0,
      active=True, driver_input=False, lat_ctl_limit=0,
      projected_curvature=0.03,
    )

  assert command.path_offset < nominal.path_offset
  assert trim.state.tracking_error < 0.0


def test_adaptive_trim_uses_openpilot_sign_headroom_at_asymmetric_wire_limits():
  trim = AdaptiveLateralTrim(enabled=True)
  nominal = LearnedLateralPathCommand(True, 0.0, -0.52, 0.003, 0.0)

  command = nominal
  for _ in range(200):
    command = trim.update(nominal, -0.02, 0.0, True, False, 0)

  assert -0.5235 <= command.path_angle <= nominal.path_angle


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
  assert command.curvature == pytest.approx(-0.00136)
  assert command.curvature_rate == pytest.approx(-0.00037)
  assert controller.update(
    path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False,
  ).valid is False


def test_learned_controller_preserves_model_c2_for_ordinary_driving():
  controller = LearnedLateralPathController()
  path = SimpleNamespace(valid=True, pathOffset=0.0, pathAngle=0.0, curvature=0.004, curvatureRate=0.0)

  command = controller.update(
    path, desired_angle_deg=20.0, actual_angle_deg=18.0, steering_rate_deg_s=2.0,
    speed_mps=25.0, eps_current_a=2.0, projected_curvature=0.0038,
    measured_curvature=0.0037, desired_curvature=0.004, lat_ctl_limit=0, active=True,
  )

  assert command.curvature == pytest.approx(path.curvature)


def test_shipped_policy_disables_unvalidated_outcome_model():
  model_path = Path(__file__).parents[1] / "ford_lateral_policy_v2.npz"
  with np.load(model_path, allow_pickle=False) as model:
    assert not np.any(model["residual.l1.weight"])
    assert not np.any(model["residual.l1.bias"])
    assert not np.any(model["residual.out.weight"])
    assert not np.any(model["residual.out.bias"])


def test_learned_controller_smoothly_releases_model_c2_for_large_maneuvers():
  path = SimpleNamespace(valid=True, pathOffset=0.0, pathAngle=0.0, curvature=0.004, curvatureRate=0.0)

  commands = []
  for desired_angle_deg in (20.0, 57.5, 100.0):
    commands.append(LearnedLateralPathController().update(
      path, desired_angle_deg=desired_angle_deg, actual_angle_deg=0.0, steering_rate_deg_s=0.0,
      speed_mps=8.0, eps_current_a=2.0, projected_curvature=0.0,
      measured_curvature=0.0, desired_curvature=0.02, lat_ctl_limit=0, active=True,
    ))

  ordinary, transition, hard = commands
  assert 0.0 < abs(transition.curvature) < abs(ordinary.curvature)
  assert hard.curvature == 0.0


def test_learned_controller_does_not_release_c2_for_fast_preview_geometry_alone():
  controller = LearnedLateralPathController()
  path = SimpleNamespace(valid=True, pathOffset=2.0, pathAngle=0.0, curvature=0.004, curvatureRate=0.0)

  command = controller.update(
    path, desired_angle_deg=20.0, actual_angle_deg=0.0, steering_rate_deg_s=0.0,
    speed_mps=8.0, eps_current_a=2.0, projected_curvature=0.0,
    measured_curvature=0.0, desired_curvature=0.02, lat_ctl_limit=0, active=True,
  )

  assert command.curvature == pytest.approx(path.curvature)


def test_learned_controller_releases_c2_when_model_c2_is_large():
  controller = LearnedLateralPathController()
  path = SimpleNamespace(valid=True, pathOffset=0.0, pathAngle=0.0, curvature=0.02, curvatureRate=0.0)

  command = controller.update(
    path, desired_angle_deg=20.0, actual_angle_deg=0.0, steering_rate_deg_s=0.0,
    speed_mps=8.0, eps_current_a=2.0, projected_curvature=0.0,
    measured_curvature=0.0, desired_curvature=0.02, lat_ctl_limit=0, active=True,
  )

  assert command.curvature == 0.0


def test_learned_controller_does_not_extend_after_projected_arrival():
  path = SimpleNamespace(
    valid=True, pathOffset=0.257308691740036, pathAngle=0.092901848256588,
    curvature=0.012884913012385368, curvatureRate=0.0010777440620586276,
  )
  args = (
    path, -54.77340316772461, -55.0, 0.0, 10.230555534362793, 1.05,
    0.0147, 0.0147, 0.014595765560680964, 0, True,
  )

  command = LearnedLateralPathController().update(*args)
  driver_gated = LearnedLateralPathController().update(*args, True)

  assert command == driver_gated


def test_learned_controller_does_not_extend_across_fast_unwind_arrival():
  path = SimpleNamespace(
    valid=True, pathOffset=0.257308691740036, pathAngle=0.092901848256588,
    curvature=0.012884913012385368, curvatureRate=0.0010777440620586276,
  )
  args = (path, -50.0, -70.0, 100.0, 10.0, 1.05, 0.01, 0.014, 0.014, 0, True)

  command = LearnedLateralPathController().update(*args)
  driver_gated = LearnedLateralPathController().update(*args, True)

  assert command == driver_gated


def test_learned_controller_honors_smoothed_projected_curvature_arrival():
  path = SimpleNamespace(
    valid=True, pathOffset=0.257308691740036, pathAngle=0.092901848256588,
    curvature=0.012884913012385368, curvatureRate=0.0010777440620586276,
  )
  # Instantaneous steering rate still looks behind, but the controller's
  # smoothed wheel projection has already crossed the desired curvature.
  args = (path, -50.0, -30.0, -10.0, 10.0, 1.05, 0.016, 0.012, 0.014, 0, True)

  command = LearnedLateralPathController().update(*args)
  driver_gated = LearnedLateralPathController().update(*args, True)

  assert command == driver_gated


def test_learned_controller_preserves_highway_behavior_outside_training_envelope():
  path = SimpleNamespace(
    valid=True, pathOffset=0.257308691740036, pathAngle=0.092901848256588,
    curvature=0.012884913012385368, curvatureRate=0.0010777440620586276,
  )
  args = (path, -55.0, -20.0, -30.0, 25.0, 1.05, 0.008, 0.006, 0.014, 0, True)

  command = LearnedLateralPathController().update(*args)
  driver_gated = LearnedLateralPathController().update(*args, True)

  assert command == driver_gated


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_learned_controller_rejects_opposite_direction_outcome_residual(tmp_path, direction):
  model_path = Path(__file__).parents[1] / "ford_lateral_policy_v2.npz"
  with np.load(model_path, allow_pickle=False) as model:
    adversarial = {name: model[name] for name in model.files}
  adversarial["residual.l1.weight"] = np.zeros_like(adversarial["residual.l1.weight"])
  adversarial["residual.l1.bias"] = np.zeros_like(adversarial["residual.l1.bias"])
  adversarial["residual.out.weight"] = np.zeros_like(adversarial["residual.out.weight"])
  # A residual with wire curvature opposite the desired steering direction.
  adversarial["residual.out.bias"] = np.full_like(adversarial["residual.out.bias"], -10.0 * direction)
  adversarial_path = tmp_path / "adversarial.npz"
  np.savez(adversarial_path, **adversarial)

  path = SimpleNamespace(
    valid=True, pathOffset=-direction * 0.257308691740036, pathAngle=-direction * 0.092901848256588,
    curvature=-direction * 0.012884913012385368, curvatureRate=-direction * 0.0010777440620586276,
  )
  args = (path, direction * 55.0, direction * 20.0, direction * 30.0,
          10.0, 1.05, -direction * 0.008, -direction * 0.006, -direction * 0.014, 0, True)

  command = LearnedLateralPathController(adversarial_path).update(*args)
  driver_gated = LearnedLateralPathController(adversarial_path).update(*args, True)

  assert command == driver_gated


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_learned_controller_keeps_mixed_residual_on_model_side_at_every_horizon(tmp_path, direction):
  model_path = Path(__file__).parents[1] / "ford_lateral_policy_v2.npz"
  with np.load(model_path, allow_pickle=False) as model:
    adversarial = {name: model[name] for name in model.files}
  adversarial["residual.l1.weight"] = np.zeros_like(adversarial["residual.l1.weight"])
  adversarial["residual.l1.bias"] = np.zeros_like(adversarial["residual.l1.bias"])
  adversarial["residual.out.weight"] = np.zeros_like(adversarial["residual.out.weight"])
  residual = np.asarray((-direction * 1.0, direction * 0.2, 0.0))
  adversarial["residual.out.bias"] = np.arctanh(residual / adversarial["residual.scales"])
  adversarial_path = tmp_path / "mixed_adversarial.npz"
  np.savez(adversarial_path, **adversarial)

  path = SimpleNamespace(
    valid=True, pathOffset=-direction * 0.257308691740036, pathAngle=-direction * 0.092901848256588,
    curvature=-direction * 0.012884913012385368, curvatureRate=-direction * 0.0010777440620586276,
  )
  args = (path, direction * 55.0, direction * 20.0, direction * 30.0,
          10.0, 1.05, -direction * 0.008, -direction * 0.006, -direction * 0.014, 0, True)
  command = LearnedLateralPathController(adversarial_path).update(*args)
  baseline = LearnedLateralPathController(adversarial_path).update(*args, True)

  model_coefficients = np.asarray((path.pathOffset, path.pathAngle, path.curvature, path.curvatureRate))
  command_coefficients = np.asarray(command.coefficients())
  baseline_coefficients = np.asarray(baseline.coefficients())
  for distance in (3.0, 5.0, 7.0, 10.0):
    weights = np.asarray((2.0 / distance**2, 2.0 / distance, 1.0, distance / 3.0))
    model_curvature = float(model_coefficients @ weights)
    command_curvature = float(command_coefficients @ weights)
    assert model_curvature * command_curvature >= 0.0
  weights_7m = np.asarray((2.0 / 49.0, 2.0 / 7.0, 1.0, 7.0 / 3.0))
  assert float(model_coefficients @ weights_7m) * float((command_coefficients - baseline_coefficients) @ weights_7m) > 0.0


def test_adaptive_trim_cannot_reverse_an_admitted_residual_arc(tmp_path):
  model_path = Path(__file__).parents[1] / "ford_lateral_policy_v2.npz"
  with np.load(model_path, allow_pickle=False) as model:
    adversarial = {name: model[name] for name in model.files}
  adversarial["residual.l1.weight"] = np.zeros_like(adversarial["residual.l1.weight"])
  adversarial["residual.l1.bias"] = np.zeros_like(adversarial["residual.l1.bias"])
  adversarial["residual.out.weight"] = np.zeros_like(adversarial["residual.out.weight"])
  residual = np.asarray((1.17683657, -0.252244421, -0.000545466214))
  adversarial["residual.out.bias"] = np.arctanh(residual / adversarial["residual.scales"])
  adversarial_path = tmp_path / "adaptive_adversarial.npz"
  np.savez(adversarial_path, **adversarial)

  controller = LearnedLateralPathController(adversarial_path)
  controller.set_adaptive_enabled(True)
  controller.adaptive_trim._gains[1] = 0.12
  path = SimpleNamespace(
    valid=True, pathOffset=0.257308691740036, pathAngle=0.092901848256588,
    curvature=0.012884913012385368, curvatureRate=0.0010777440620586276,
  )
  command = controller.update(
    path, -54.77340316772461, -23.0, -44.0, 10.230555534362793, 1.05,
    0.00945987737547583, 0.006128934637632228, 0.014595765560680964, 0, True,
  )
  model_coefficients = np.asarray((path.pathOffset, path.pathAngle, path.curvature, path.curvatureRate))
  command_coefficients = np.asarray(command.coefficients())
  for distance in (3.0, 5.0, 7.0, 10.0):
    weights = np.asarray((2.0 / distance**2, 2.0 / distance, 1.0, distance / 3.0))
    assert float(model_coefficients @ weights) * float(command_coefficients @ weights) >= 0.0


def test_learned_controller_rejects_unversioned_artifact(tmp_path):
  model_path = Path(__file__).parents[1] / "ford_lateral_policy_v2.npz"
  with np.load(model_path, allow_pickle=False) as model:
    unversioned = {name: model[name] for name in model.files if name != "version"}
  unversioned_path = tmp_path / "unversioned.npz"
  np.savez(unversioned_path, **unversioned)

  with pytest.raises(ValueError, match="version"):
    LearnedLateralPathController(unversioned_path)


def test_learned_controller_toggle_enables_adaptive_trim_around_nominal_policy():
  nominal_controller = LearnedLateralPathController()
  adaptive_controller = LearnedLateralPathController()
  adaptive_controller.set_adaptive_enabled(True)
  path = SimpleNamespace(valid=True, pathOffset=0.59, pathAngle=0.031, curvature=-0.00137, curvatureRate=-0.00102)

  args = (path, -100.0, 0.0, 0.0, 25.0, 8.0, 0.0, 0.0, -0.02, 0, True)
  nominal = nominal_controller.update(*args)
  adaptive = nominal
  for _ in range(200):
    adaptive = adaptive_controller.update(*args)

  assert abs(adaptive.path_angle) > abs(nominal.path_angle)
  assert adaptive.curvature == nominal.curvature
  adaptive_controller.set_adaptive_enabled(False)
  assert adaptive_controller.update(*args) == nominal


def test_learned_controller_clears_adapting_state_while_inactive_without_forgetting_gain():
  controller = LearnedLateralPathController()
  controller.set_adaptive_enabled(True)
  path = SimpleNamespace(valid=True, pathOffset=0.59, pathAngle=0.031, curvature=-0.00137, curvatureRate=-0.00102)
  args = (path, -100.0, 0.0, 0.0, 25.0, 8.0, 0.0, 0.0, -0.02, 0, True)

  for _ in range(200):
    controller.update(*args)
  assert controller.adaptive_state.adapting
  learned_command = controller.update(*args, True)

  assert not controller.update(path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False).valid
  assert controller.adaptive_state.enabled
  assert not controller.adaptive_state.adapting
  assert controller.adaptive_state.tracking_error == 0.0
  assert controller.update(*args, True) == learned_command
