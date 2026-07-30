import math
from types import SimpleNamespace
import unittest

from opendbc.car.ford.lateral_path_projector import (
  FEEDBACK_PREVIEW_LIMIT,
  ProjectedLatControlPath,
  _curvature,
  _retain_feedback_preview,
)


def model(path_offset: float, path_angle: float, curvature: float = 0.0, curvature_rate: float = 0.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=path_offset,
    pathAngle=path_angle,
    curvature=curvature,
    curvatureRate=curvature_rate,
  )


def equivalent_curvature(command, distance: float) -> float:
  path_offset = getattr(command, "path_offset", getattr(command, "pathOffset", 0.0))
  path_angle = getattr(command, "path_angle", getattr(command, "pathAngle", 0.0))
  curvature_rate = getattr(command, "curvature_rate", getattr(command, "curvatureRate", 0.0))
  y = path_offset + path_angle * distance + 0.5 * command.curvature * distance ** 2 + \
      curvature_rate * distance ** 3 / 6.0
  return 2.0 * y / distance ** 2


def test_feasible_steady_model_is_reproduced_by_c2():
  controller = ProjectedLatControlPath()
  delivered_curvature = 0.015
  target = model(
    0.5 * delivered_curvature * 7.0 ** 2,
    delivered_curvature * 7.0,
    delivered_curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      target, delivered_curvature, 7.0, True, False,
      desired_angle_curvature=delivered_curvature,
    )

  assert command is not None
  for distance in (3.0, 7.0, 15.0, 30.0):
    assert abs(equivalent_curvature(command, distance) - delivered_curvature) < 1e-6


def test_clipped_coefficients_remain_bounded_and_directionally_coherent():
  controller = ProjectedLatControlPath()
  target = model(-2.7, -0.60, -0.05, -0.002)

  command = None
  for _ in range(100):
    command = controller.update(target, -0.03, 7.0, True, False)

  assert command is not None
  assert -4.61 <= command.path_offset <= 4.60
  assert -0.475 <= command.path_angle <= 0.497
  assert -0.02 <= command.curvature <= 0.02
  assert -0.001024 <= command.curvature_rate <= 0.001023
  assert equivalent_curvature(command, 7.0) < 0.0


def test_large_turn_flushes_c2_and_projects_its_path_into_other_coefficients():
  controller = ProjectedLatControlPath()
  target = model(0.8, 0.2, 0.015)

  command = None
  for _ in range(100):
    command = controller.update(target, 0.012, 7.0, True, False)

  assert command is not None
  assert command.curvature == 0.0
  assert command.path_offset > target.pathOffset or command.path_angle > target.pathAngle


def test_meaningful_model_exit_cannot_project_to_the_opposite_direction():
  controller = ProjectedLatControlPath()
  target = model(-0.0916, -0.0217, -0.0017, 0.0005)

  command = controller.update(
    target, -0.0046, 11.54, True, False,
    desired_angle_curvature=-0.0017,
  )

  assert equivalent_curvature(target, 7.0) < -0.005
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_projected_arrival_removes_only_c0_c1_correction():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  behind_controller = ProjectedLatControlPath()
  arrived_controller = ProjectedLatControlPath()

  behind_command = None
  arrived_command = None
  for _ in range(100):
    behind_command = behind_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )
    arrived_command = arrived_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.016,
      desired_angle_curvature=desired_curvature,
    )

  assert behind_command is not None
  assert arrived_command is not None
  assert behind_command.path_offset > arrived_command.path_offset
  assert behind_command.path_angle > arrived_command.path_angle
  assert abs(arrived_command.path_offset - target.pathOffset) < 1e-9
  assert abs(arrived_command.path_angle - target.pathAngle) < 1e-9


def test_spatial_preview_keeps_full_authority_while_wheel_is_behind():
  model_curvature = 0.07
  desired_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_curvature,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  assert command.path_offset > target.pathOffset
  assert command.path_angle >= target.pathAngle


def test_rising_spatial_turn_stops_pinned_preview_after_wheel_passes_lagging_action():
  controller = ProjectedLatControlPath()
  spatial_curvature = 0.04
  action_curvature = 0.002
  rising_turn = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    action_curvature,
    0.005,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      rising_turn, 0.01, 7.0, True, False,
      projected_measured_curvature=0.01,
      desired_angle_curvature=action_curvature,
    )

  assert command is not None
  assert 0.0 < equivalent_curvature(command, 7.0) <= action_curvature + 0.006001
  assert command.curvature == 0.0
  assert 0.0 < command.curvature_rate < 0.001023


def test_spatial_preview_is_symmetric_for_right_turns():
  # Keep the raw polynomial inside Ford's intentionally asymmetric signal
  # bounds so this isolates controller symmetry from DBC clipping.
  model_curvature = 0.04
  desired_curvature = 0.015
  left_target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_curvature,
    0.0005,
  )
  right_target = model(
    -left_target.pathOffset,
    -left_target.pathAngle,
    -left_target.curvature,
    -left_target.curvatureRate,
  )
  left_controller = ProjectedLatControlPath()
  right_controller = ProjectedLatControlPath()

  left = None
  right = None
  for _ in range(100):
    left = left_controller.update(
      left_target, desired_curvature, 7.0, True, False,
      projected_measured_curvature=desired_curvature,
      desired_angle_curvature=desired_curvature,
    )
    right = right_controller.update(
      right_target, -desired_curvature, 7.0, True, False,
      projected_measured_curvature=-desired_curvature,
      desired_angle_curvature=-desired_curvature,
    )

  assert left is not None
  assert right is not None
  assert right.valid == left.valid
  for left_coefficient, right_coefficient in zip(left.coefficients(), right.coefficients(), strict=True):
    assert abs(right_coefficient + left_coefficient) < 1e-12


def test_confirmed_low_speed_spatial_onset_uses_preview_while_wheel_is_behind():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    desired_curvature = direction * 0.006
    spatial_curvature = direction * 0.015
    target = model(
      0.5 * spatial_curvature * 7.0 ** 2,
      spatial_curvature * 7.0,
      desired_curvature,
      direction * 0.001,
    )

    command = None
    for _ in range(100):
      command = controller.update(
        target, 0.0, 5.0, True, False,
        projected_measured_curvature=0.0,
        desired_angle_curvature=desired_curvature,
      )

    assert command is not None
    assert direction * command.path_offset > 0.0
    assert direction * command.path_angle > 0.0
    assert 0.0 < direction * command.curvature < abs(desired_curvature)
    assert direction * command.curvature_rate > 0.0


def test_unconfirmed_low_speed_spatial_slope_stays_on_c2():
  cases = (
    (0.006, 0.0007),
    (0.010, 0.0010),
  )
  for desired_curvature, curvature_rate in cases:
    controller = ProjectedLatControlPath()
    spatial_curvature = 0.015
    target = model(
      0.5 * spatial_curvature * 7.0 ** 2,
      spatial_curvature * 7.0,
      desired_curvature,
      curvature_rate,
    )

    command = None
    for _ in range(100):
      command = controller.update(
        target, 0.0, 5.0, True, False,
        projected_measured_curvature=0.0,
        desired_angle_curvature=desired_curvature,
      )

    assert command is not None
    assert command.path_offset == 0.0
    assert command.path_angle == 0.0
    assert command.curvature == desired_curvature
    assert command.curvature_rate == 0.0


def test_confirmed_spatial_onset_is_removed_immediately_at_projected_arrival():
  controller = ProjectedLatControlPath()
  desired_curvature = 0.006
  spatial_curvature = 0.015
  target = model(
    0.5 * spatial_curvature * 7.0 ** 2,
    spatial_curvature * 7.0,
    desired_curvature,
    0.001,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.0, 5.0, True, False,
      projected_measured_curvature=0.0,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  assert command.path_offset > 0.0
  arrived = controller.update(
    target, 0.0, 5.0, True, False,
    projected_measured_curvature=desired_curvature,
    desired_angle_curvature=desired_curvature,
  )

  assert arrived.path_offset == 0.0
  assert arrived.path_angle == 0.0
  assert arrived.curvature > command.curvature
  assert arrived.curvature_rate == 0.0


def test_continuing_model_preview_extends_only_c0_after_current_angle_arrival():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.002,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert command.path_offset > target.pathOffset
  assert abs(command.path_angle - target.pathAngle) < 1e-9
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.001023

  bounded_preview_arrived = controller.update(
    target, 0.01, 7.0, True, False,
    projected_measured_curvature=desired_angle_curvature + 0.006,
    desired_angle_curvature=desired_angle_curvature,
  )

  assert abs(bounded_preview_arrived.path_offset - target.pathOffset) < 1e-9
  assert abs(bounded_preview_arrived.path_angle - target.pathAngle) < 1e-9


def test_continuing_model_preview_is_bounded_after_wheel_passes_angle_corridor():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.002,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, desired_angle_curvature + 0.007, 7.0, True, False,
      projected_measured_curvature=desired_angle_curvature + 0.009,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert 0.0 < equivalent_curvature(command, 7.0) <= desired_angle_curvature + 0.006001


def test_available_c3_does_not_spill_continuation_into_c0():
  model_curvature = 0.04
  desired_angle_curvature = 0.015
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    desired_angle_curvature,
    0.0005,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.01, 7.0, True, False,
      projected_measured_curvature=0.02,
      desired_angle_curvature=desired_angle_curvature,
    )

  assert command is not None
  assert abs(command.path_offset - target.pathOffset) < 1e-9
  assert abs(command.path_angle - target.pathAngle) < 1e-9
  assert command.curvature_rate == target.curvatureRate


def test_pscm_envelope_reallocates_outward_c3_into_c0_while_wheel_is_behind():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  controllers = [ProjectedLatControlPath() for _ in range(3)]
  commands = [None, None, None]

  for _ in range(100):
    for limit_status, controller in enumerate(controllers):
      commands[limit_status] = controller.update(
        target, 0.005, 7.0, True, False,
        projected_measured_curvature=0.005,
        desired_angle_curvature=desired_curvature,
        lat_ctl_limit=limit_status,
      )

  clear, close, reached = commands
  assert clear is not None
  assert close is not None
  assert reached is not None
  assert clear.path_offset < close.path_offset < reached.path_offset
  assert clear.curvature_rate > close.curvature_rate > reached.curvature_rate
  assert abs(close.curvature_rate - 0.5 * clear.curvature_rate) < 1e-12
  assert abs(reached.curvature_rate) < 1e-12
  assert equivalent_curvature(clear, 3.0) < equivalent_curvature(close, 3.0) < equivalent_curvature(reached, 3.0)
  assert equivalent_curvature(clear, 7.0) < equivalent_curvature(close, 7.0) < equivalent_curvature(reached, 7.0)
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(close, 15.0)) < 1e-9
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(reached, 15.0)) < 1e-9


def test_pscm_envelope_does_not_reallocate_c3_after_projected_arrival():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  baseline_controller = ProjectedLatControlPath()
  controller = ProjectedLatControlPath()

  baseline = None
  reallocated = None
  for _ in range(100):
    baseline = baseline_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.016,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    reallocated = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=2,
    )

  arrived = controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.016,
    desired_angle_curvature=desired_curvature,
    lat_ctl_limit=2,
  )

  assert baseline is not None
  assert reallocated is not None
  assert reallocated != baseline
  assert arrived == baseline


def test_pscm_driver_limit_does_not_add_path_authority():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  clear_controller = ProjectedLatControlPath()
  driver_limit_controller = ProjectedLatControlPath()

  clear = None
  driver_limited = None
  for _ in range(100):
    clear = clear_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    driver_limited = driver_limit_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=3,
    )

  assert clear == driver_limited


def test_pscm_envelope_reallocation_is_symmetric_for_right_turns():
  desired_curvature = -0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    -0.003,
  )
  clear_controller = ProjectedLatControlPath()
  reached_controller = ProjectedLatControlPath()

  clear = None
  reached = None
  for _ in range(100):
    clear = clear_controller.update(
      target, -0.005, 7.0, True, False,
      projected_measured_curvature=-0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=0,
    )
    reached = reached_controller.update(
      target, -0.005, 7.0, True, False,
      projected_measured_curvature=-0.005,
      desired_angle_curvature=desired_curvature,
      lat_ctl_limit=2,
    )

  assert clear is not None
  assert reached is not None
  assert abs(reached.path_offset) > abs(clear.path_offset)
  assert abs(reached.curvature_rate) < abs(clear.curvature_rate)
  assert abs(equivalent_curvature(clear, 15.0) - equivalent_curvature(reached, 15.0)) < 1e-9


def test_desired_angle_shortfall_extends_c0_c1_beyond_model_geometry():
  model_curvature = 0.008
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    model_curvature,
    0.003,
  )
  model_controller = ProjectedLatControlPath()
  deeper_controller = ProjectedLatControlPath()

  model_command = None
  deeper_command = None
  for _ in range(100):
    model_command = model_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=model_curvature,
    )
    deeper_command = deeper_controller.update(
      target, 0.003, 7.0, True, False,
      projected_measured_curvature=0.003,
      desired_angle_curvature=0.015,
    )

  assert model_command is not None
  assert deeper_command is not None
  assert deeper_command.path_offset > model_command.path_offset
  assert deeper_command.path_angle > model_command.path_angle


def test_projected_gate_preserves_model_correction_while_both_wheel_estimates_are_behind():
  model_curvature = 0.015
  desired_curvature = 0.010
  measured_curvature = 0.005
  tracking_correction = model_curvature - measured_curvature - 0.0005
  target = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    model_curvature,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, measured_curvature, 7.0, True, False,
      projected_measured_curvature=measured_curvature,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  assert abs(command.path_offset - 0.5 * (model_curvature + tracking_correction) * 7.0 ** 2) < 1e-9
  assert abs(command.path_angle - (model_curvature + tracking_correction) * 7.0) < 1e-9


def test_projected_crossing_drops_correction_immediately_without_reversing_model():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=desired_curvature,
    )

  assert command is not None
  arrived = controller.update(
    target, 0.005, 7.0, True, False,
    projected_measured_curvature=0.016,
    desired_angle_curvature=desired_curvature,
  )

  assert arrived.path_offset < command.path_offset
  assert arrived.path_angle < command.path_angle
  assert abs(arrived.path_offset - target.pathOffset) < 1e-9
  assert abs(arrived.path_angle - target.pathAngle) < 1e-9


def test_projected_arrival_tapers_c0_c1_correction_without_a_command_step():
  desired_curvature = 0.015
  target = model(
    0.5 * desired_curvature * 7.0 ** 2,
    desired_curvature * 7.0,
    desired_curvature,
    0.003,
  )
  just_outside_controller = ProjectedLatControlPath()
  just_inside_controller = ProjectedLatControlPath()

  just_outside = None
  just_inside = None
  for _ in range(100):
    just_outside = just_outside_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.01449,
      desired_angle_curvature=desired_curvature,
    )
    just_inside = just_inside_controller.update(
      target, 0.005, 7.0, True, False,
      projected_measured_curvature=0.01451,
      desired_angle_curvature=desired_curvature,
    )

  assert just_outside is not None
  assert just_inside is not None
  assert equivalent_curvature(just_outside, 7.0) >= equivalent_curvature(just_inside, 7.0)
  assert equivalent_curvature(just_outside, 7.0) - equivalent_curvature(just_inside, 7.0) < 0.001


def test_measured_wheel_and_desired_angle_reject_stale_opposing_geometry():
  controller = ProjectedLatControlPath()
  stale_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, -0.0004)

  command = controller.update(
    stale_geometry, 0.01, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) <= 0.0


def test_c3_that_continues_turn_preserves_preview_through_action_conflict():
  controller = ProjectedLatControlPath()
  continuing_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, 0.003)

  command = controller.update(
    continuing_geometry, 0.01, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) > 0.0


def test_opposing_action_does_not_discard_model_preview_before_wheel_follows_it():
  controller = ProjectedLatControlPath()
  entering_geometry = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, -0.004, 0.003)

  command = controller.update(
    entering_geometry, -0.003, 7.0, True, False,
    desired_angle_curvature=-0.004,
  )

  assert equivalent_curvature(command, 7.0) > 0.0


def test_opposing_action_does_not_block_c0_c1_correction_toward_desired_angle():
  model_curvature = 0.015
  entering_geometry = model(
    0.5 * model_curvature * 7.0 ** 2,
    model_curvature * 7.0,
    -0.004,
    0.003,
  )
  controller = ProjectedLatControlPath()

  command = None
  for _ in range(100):
    command = controller.update(
      entering_geometry, 0.005, 7.0, True, False,
      projected_measured_curvature=0.005,
      desired_angle_curvature=model_curvature,
    )

  assert command is not None
  assert command.path_offset > entering_geometry.pathOffset
  assert command.path_angle > entering_geometry.pathAngle


def test_medium_curve_allocates_c2_once_without_opposing_preview_coefficients():
  controller = ProjectedLatControlPath()
  curvature = 0.005
  target = model(0.5 * curvature * 7.0 ** 2, curvature * 7.0, curvature)

  command = None
  for _ in range(100):
    command = controller.update(
      target, curvature, 7.0, True, False,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.curvature > 0.0
  assert command.path_offset >= 0.0
  assert command.path_angle >= 0.0


def test_overtracking_reduces_authority_without_zeroing_model_path():
  target = model(0.5 * 0.015 * 7.0 ** 2, 0.015 * 7.0, 0.015, 0.003)
  behind_controller = ProjectedLatControlPath()
  beyond_controller = ProjectedLatControlPath()

  behind = None
  beyond = None
  for _ in range(100):
    behind = behind_controller.update(
      target, 0.005, 7.0, True, False,
      desired_angle_curvature=0.015,
    )
    beyond = beyond_controller.update(
      target, 0.025, 7.0, True, False,
      desired_angle_curvature=0.015,
    )

  assert behind is not None
  assert beyond is not None
  assert 0.0 < equivalent_curvature(beyond, 7.0) < equivalent_curvature(behind, 7.0)


def test_delivered_gentle_path_uses_c2_without_duplicate_preview_terms():
  controller = ProjectedLatControlPath()
  gentle = model(0.5 * 0.003 * 7.0 ** 2, 0.003 * 15.0, 0.003)

  command = None
  for _ in range(100):
    command = controller.update(gentle, 0.003, 15.0, True, False)

  assert command is not None
  assert abs(command.path_offset) < 1e-6
  assert abs(command.path_angle) < 1e-6
  assert abs(command.curvature - 0.003) < 1e-6


def test_spatially_steady_high_curvature_stays_on_c2():
  controller = ProjectedLatControlPath()
  curvature = 0.012
  speed = 10.0
  steady_curve = model(
    0.5 * curvature * 7.0 ** 2,
    curvature * speed,
    curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      steady_curve, curvature, speed, True, False,
      projected_measured_curvature=curvature,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature == curvature
  assert command.curvature_rate == 0.0


def test_gentle_path_uses_only_reversible_c2_despite_noisy_preview():
  controller = ProjectedLatControlPath()
  gentle = model(
    0.5 * 0.0025 * 7.0 ** 2,
    0.0025 * 25.0,
    0.002,
    -0.0001,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      gentle, 0.002, 25.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.002,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature == gentle.curvature
  assert command.curvature_rate == 0.0


def test_gentle_c2_anchor_is_not_changed_by_measured_wheel_disturbance():
  controller = ProjectedLatControlPath()
  gentle = model(
    0.5 * 0.0025 * 7.0 ** 2,
    0.0025 * 25.0,
    0.002,
    -0.0001,
  )

  for _ in range(100):
    controller.update(
      gentle, 0.002, 25.0, True, False,
      projected_measured_curvature=0.002,
      desired_angle_curvature=0.002,
    )

  disturbed = controller.update(
    gentle, 0.0065, 25.0, True, False,
    projected_measured_curvature=-0.001,
    desired_angle_curvature=0.002,
  )

  assert disturbed.path_offset == 0.0
  assert disturbed.path_angle == 0.0
  assert disturbed.curvature == gentle.curvature
  assert disturbed.curvature_rate == 0.0


def test_spatial_slope_crossfades_once_from_c2_to_full_polynomial():
  commands = []
  curvature = 0.003
  for maneuver_demand in (0.003, 0.0045, 0.006):
    controller = ProjectedLatControlPath()
    target = model(
      0.5 * curvature * 7.0 ** 2,
      curvature * 7.0,
      curvature,
      maneuver_demand * 3.0 / 7.0,
    )
    command = None
    for _ in range(100):
      command = controller.update(
        target, curvature, 7.0, True, False,
        desired_angle_curvature=curvature,
      )
    commands.append(command)

  assert all(command is not None for command in commands)
  gentle, transition, full = commands
  assert abs(gentle.curvature - 0.003) < 1e-12
  assert abs(gentle.path_offset) < 1e-12
  assert abs(gentle.path_angle) < 1e-12
  assert abs(transition.curvature - 0.0015) < 1e-9
  assert transition.path_offset > 0.0
  assert transition.path_angle > 0.0
  assert full.curvature == 0.0
  assert full.path_offset > transition.path_offset
  assert full.path_angle > transition.path_angle


def test_single_preview_observation_cannot_pull_ordinary_c2_into_polynomial_transition():
  controller = ProjectedLatControlPath()
  curvature = 0.0017
  offset_curvature = 0.0047
  angle_curvature = 0.0025
  target = model(
    0.5 * offset_curvature * 7.0 ** 2,
    angle_curvature * 12.0,
    curvature,
  )

  command = None
  for _ in range(100):
    command = controller.update(
      target, curvature, 12.0, True, False,
      projected_measured_curvature=curvature,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert abs(command.curvature - curvature) < 1e-9


def test_meaningful_c3_preview_can_leave_c2_baseband():
  controller = ProjectedLatControlPath()
  target = model(0.0, 0.0, 0.001, 0.0005)

  command = None
  for _ in range(100):
    command = controller.update(
      target, 0.001, 25.0, True, False,
      projected_measured_curvature=0.001,
      desired_angle_curvature=0.001,
    )

  assert command is not None
  assert 0.0 < command.curvature < target.curvature
  assert command.curvature_rate > 0.0


def test_gentle_steady_curve_does_not_reset_when_wheel_temporarily_overtracks():
  controller = ProjectedLatControlPath()
  curvature = 0.003
  target = model(0.5 * curvature * 7.0 ** 2, curvature * 7.0, curvature)

  command = None
  for _ in range(20):
    command = controller.update(
      target, curvature, 7.0, True, False,
      desired_angle_curvature=curvature,
    )

  assert command is not None
  settled_curvature = equivalent_curvature(command, 7.0)
  disturbed = controller.update(
    target, 0.0065, 7.0, True, False,
    desired_angle_curvature=curvature,
  )

  assert abs(settled_curvature - curvature) < 1e-9
  assert equivalent_curvature(disturbed, 7.0) >= 0.5 * settled_curvature
  assert disturbed.path_offset >= 0.0
  assert disturbed.path_angle >= 0.0
  assert disturbed.curvature > 0.0


def test_zero_unwind_target_cannot_manufacture_opposing_preview():
  controller = ProjectedLatControlPath()
  target = model(-0.0245, -0.007, -0.001)

  command = None
  for _ in range(10):
    command = controller.update(
      target, -0.0031, 7.0, True, False,
      desired_angle_curvature=-0.001,
    )

  assert command is not None
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_c3_unwind_waits_while_wheel_undertracks_desired_angle():
  controller = ProjectedLatControlPath()
  target = model(0.5, 0.1, 0.01, -0.0004)

  command = controller.update(
    target, 0.006, 7.0, True, False,
    projected_measured_curvature=0.006,
    desired_angle_curvature=0.01,
  )

  assert command.curvature_rate == 0.0


def test_gentle_c3_unwind_stops_after_projected_wheel_crosses_target():
  controller = ProjectedLatControlPath()
  target = model(-0.0245, -0.007, -0.00175, 0.00052)

  command = controller.update(
    target, -0.00175, 7.0, True, False,
    projected_measured_curvature=0.0005,
    desired_angle_curvature=-0.00175,
  )

  assert command.curvature_rate == 0.0


def test_projection_constraints_cannot_flip_delivered_model_geometry():
  controller = ProjectedLatControlPath()
  target = model(-0.1324408266, -0.0564401015, 0.0084251088, -0.0005080627)

  command = controller.update(
    target, -0.005, 10.0, True, False,
    projected_measured_curvature=-0.005,
    desired_angle_curvature=-0.005,
  )

  assert equivalent_curvature(target, 7.0) < -0.01
  assert equivalent_curvature(command, 7.0) <= 0.0


def test_driver_override_projects_the_delivered_wheel_path():
  controller = ProjectedLatControlPath()

  command = controller.update(model(1.0, 0.2, 0.02, 0.001), -0.01, 10.0, True, True)

  assert command.path_offset == 0.5 * -0.01 * 7.0 ** 2
  assert command.path_angle == -0.01 * 10.0
  assert command.curvature == 0.0
  assert command.curvature_rate == 0.0


def test_all_coefficients_attack_without_software_rate_limits_and_release_is_immediate():
  controller = ProjectedLatControlPath()

  attack = controller.update(model(4.0, 0.4, 0.02, 0.001), 0.0, 7.0, True, False)
  release = controller.update(model(0.0, 0.0), 0.0, 7.0, True, False)

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497
  assert attack.curvature_rate == 0.001
  assert release.path_offset == 0.0
  assert release.path_angle == 0.0
  assert release.curvature == 0.0
  assert release.curvature_rate == 0.0


def test_c2_attack_is_not_software_rate_limited():
  controller = ProjectedLatControlPath()

  attack = controller.update(
    model(0.0, 0.0, 0.002), 0.0, 7.0, True, False,
    desired_angle_curvature=0.002,
  )

  assert attack.curvature == 0.002


def test_large_undertracking_maneuver_is_not_software_rate_limited():
  controller = ProjectedLatControlPath()
  target = model(4.0, 0.4, 0.02, 0.001)

  attack = controller.update(
    target, 0.0, 7.0, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.02,
  )

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497


def test_projected_arrival_does_not_reintroduce_c0_c1_attack_limit():
  controller = ProjectedLatControlPath()
  target = model(4.0, 0.4, 0.02, 0.001)

  attack = controller.update(
    target, 0.015, 7.0, True, False,
    projected_measured_curvature=0.021,
    desired_angle_curvature=0.02,
  )

  assert 0.18375 < attack.path_offset <= 4.60
  assert 0.0525 < attack.path_angle <= 0.497


def test_c2_baseband_uses_small_c0_trim_while_wheel_is_behind_desired_angle():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.049, direction * 0.014, direction * 0.002)

    command = controller.update(
      target, 0.0, 7.0, True, False,
      projected_measured_curvature=0.0,
      desired_angle_curvature=direction * 0.004,
    )

    assert command.curvature == target.curvature
    assert 0.0 < direction * command.path_offset < 0.1
    assert command.path_angle == 0.0
    assert abs(equivalent_curvature(command, 7.0) - direction * 0.00375) < 1e-12


def test_tracking_extension_remains_available_through_polynomial_handoff():
  controller = ProjectedLatControlPath()
  target = model(-0.2, -0.05, -0.01, -0.0015)

  command = controller.update(
    target, 0.0, 7.0, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=-0.03,
  )

  delivered_curvature = equivalent_curvature(command, 7.0)
  assert -0.03 < delivered_curvature <= -0.02


def test_pinned_outward_preview_cannot_keep_turning_after_wheel_passes_desired_angle():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    # Segment 25: the spatial path asks for a future right turn while the action
    # target is nearly straight and both wheel estimates are already beyond it.
    target = model(
      direction * 2.885494,
      direction * 0.558042,
      direction * 0.00015,
      direction * 0.010035,
    )

    command = controller.update(
      target, direction * 0.05, 3.7, True, False,
      projected_measured_curvature=direction * 0.06,
      desired_angle_curvature=direction * 0.00015,
      lat_ctl_limit=2,
    )

    delivered_curvature = direction * equivalent_curvature(command, 7.0)
    assert 0.0 < delivered_curvature <= 0.006151


def test_pinned_outward_preview_keeps_full_authority_before_wheel_arrives():
  controller = ProjectedLatControlPath()
  target = model(2.885494, 0.558042, 0.00015, 0.010035)

  command = controller.update(
    target, 0.0, 3.7, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.00015,
    lat_ctl_limit=2,
  )

  assert equivalent_curvature(command, 7.0) > 0.25


def test_pinned_outward_preview_relatches_when_desired_moves_beyond_wheel():
  controller = ProjectedLatControlPath()
  target = model(2.885494, 0.558042, 0.07, 0.010035)

  command = controller.update(
    target, 0.05, 3.7, True, False,
    projected_measured_curvature=0.06,
    desired_angle_curvature=0.07,
    lat_ctl_limit=2,
  )

  assert equivalent_curvature(command, 7.0) > 0.25


def test_unwind_sign_c3_is_not_reduced_after_wheel_passes_desired_angle():
  controller = ProjectedLatControlPath()
  target = model(2.885494, 0.558042, 0.00015, -0.010035)

  command = controller.update(
    target, 0.05, 3.7, True, False,
    projected_measured_curvature=0.06,
    desired_angle_curvature=0.00015,
    lat_ctl_limit=2,
  )

  assert equivalent_curvature(command, 7.0) > 0.25
  assert command.curvature_rate < 0.0


def test_c2_baseband_trim_is_removed_when_projected_wheel_arrives():
  for direction in (-1.0, 1.0):
    controller = ProjectedLatControlPath()
    target = model(direction * 0.049, direction * 0.014, direction * 0.002)

    command = controller.update(
      target, 0.0, 7.0, True, False,
      projected_measured_curvature=direction * 0.004,
      desired_angle_curvature=direction * 0.004,
    )

    assert command.path_offset == 0.0
    assert command.path_angle == 0.0
    assert command.curvature == target.curvature


def test_c2_baseband_trim_requires_model_and_desired_angle_to_agree():
  controller = ProjectedLatControlPath()
  target = model(-0.049, -0.014, -0.002)

  command = controller.update(
    target, 0.0, 7.0, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.004,
  )

  assert command.path_offset == 0.0
  assert command.path_angle == 0.0
  assert command.curvature == target.curvature


def test_clipped_c3_authority_moves_into_c0_only_while_wheel_is_behind():
  for direction, curvature_rate_limit in ((-1.0, -0.001024), (1.0, 0.001023)):
    target = model(0.0, 0.0, 0.0, direction * 0.003)
    behind_controller = ProjectedLatControlPath()
    arrived_controller = ProjectedLatControlPath()

    behind = behind_controller.update(
      target, 0.0, 7.0, True, False,
      projected_measured_curvature=0.0,
      desired_angle_curvature=direction * 0.02,
    )
    arrived = arrived_controller.update(
      target, 0.0, 7.0, True, False,
      projected_measured_curvature=direction * 0.021,
      desired_angle_curvature=direction * 0.02,
    )

    assert behind.curvature_rate == curvature_rate_limit
    assert direction * behind.path_offset > 0.0
    assert abs(equivalent_curvature(behind, 7.0)) > abs(equivalent_curvature(arrived, 7.0))
    assert arrived.path_offset == 0.0


def test_controller_is_stateless_and_order_independent():
  target_a = model(0.8, 0.2, 0.015, 0.003)
  target_b = model(-0.4, -0.1, -0.008, -0.001)
  controller = ProjectedLatControlPath()

  first = controller.update(
    target_b, -0.003, 9.0, True, False,
    projected_measured_curvature=-0.004,
    desired_angle_curvature=-0.012,
    lat_ctl_limit=2,
  )
  controller.update(
    target_a, 0.02, 4.0, True, False,
    projected_measured_curvature=0.03,
    desired_angle_curvature=0.005,
    lat_ctl_limit=1,
  )
  after_unrelated_call = controller.update(
    target_b, -0.003, 9.0, True, False,
    projected_measured_curvature=-0.004,
    desired_angle_curvature=-0.012,
    lat_ctl_limit=2,
  )

  assert after_unrelated_call == first
  assert ProjectedLatControlPath().update(
    target_b, -0.003, 9.0, True, False,
    projected_measured_curvature=-0.004,
    desired_angle_curvature=-0.012,
    lat_ctl_limit=2,
  ) == first


def test_inactive_invalid_and_nonfinite_inputs_are_safe():
  controller = ProjectedLatControlPath()
  inactive = controller.update(model(1.0, 1.0, 1.0, 1.0), float("nan"), float("inf"), False, False)
  assert inactive.coefficients() == (0.0, 0.0, 0.0, 0.0)
  assert not inactive.valid

  invalid = SimpleNamespace(
    valid=False,
    curvature=float("nan"),
  )
  command = controller.update(
    invalid, float("nan"), float("-inf"), True, False,
    projected_measured_curvature=float("inf"),
    desired_angle_curvature=float("nan"),
    lat_ctl_limit=99,
  )
  assert not command.valid
  assert all(math.isfinite(value) for value in command.coefficients())
  assert -4.61 <= command.path_offset <= 4.60
  assert -0.475 <= command.path_angle <= 0.497
  assert -0.02 <= command.curvature <= 0.02
  assert -0.001024 <= command.curvature_rate <= 0.001023


def test_pinned_preview_waits_until_both_wheel_estimates_pass_desired():
  target = model(2.885494, 0.558042, 0.00015, 0.010035)
  controller = ProjectedLatControlPath()

  one_past = controller.update(
    target, 0.05, 3.7, True, False,
    projected_measured_curvature=0.0,
    desired_angle_curvature=0.00015,
    lat_ctl_limit=2,
  )
  both_past = controller.update(
    target, 0.05, 3.7, True, False,
    projected_measured_curvature=0.06,
    desired_angle_curvature=0.00015,
    lat_ctl_limit=2,
  )

  assert equivalent_curvature(one_past, 7.0) > equivalent_curvature(both_past, 7.0) > 0.0


class TestRoute79Regressions(unittest.TestCase):
  def test_unresolved_desired_direction_cannot_be_cancelled_by_preview(self):
    # Route 79 segment 49: C2, C3 at 7 m, and desired angle request a right
    # turn while C0/C1 still describe the prior left. With both wheel estimates
    # behind, the preview terms may not cancel the current pull.
    desired_curvature = 0.0138
    controller = ProjectedLatControlPath()
    target = model(-0.37, -0.09, 0.0126, -0.00027)

    command = controller.update(
      target, -0.0106, 4.7, True, False,
      projected_measured_curvature=-0.0110,
      desired_angle_curvature=desired_curvature,
    )

    self.assertGreaterEqual(
      equivalent_curvature(command, 7.0),
      desired_curvature - 0.0005,
    )

  def test_direction_ownership_is_continuous_at_local_curvature_reversal(self):
    desired_curvature = 0.0138
    commands = []
    for curvature_rate in (-0.0018001, -0.0017999):
      commands.append(ProjectedLatControlPath().update(
        model(-0.37, -0.09, 0.0126, curvature_rate),
        -0.0106,
        4.7,
        True,
        False,
        projected_measured_curvature=-0.0110,
        desired_angle_curvature=desired_curvature,
      ))

    difference = equivalent_curvature(commands[1], 7.0) - \
      equivalent_curvature(commands[0], 7.0)
    self.assertGreaterEqual(difference, 0.0)
    self.assertLess(difference, 0.0001)

  def test_strong_opposing_model_preview_keeps_ownership(self):
    desired_curvature = 0.0138
    command = ProjectedLatControlPath().update(
      model(-0.8, -0.2, 0.0126, -0.00027),
      -0.0106,
      4.7,
      True,
      False,
      projected_measured_curvature=-0.0110,
      desired_angle_curvature=desired_curvature,
    )

    self.assertLess(equivalent_curvature(command, 7.0), 0.0)

  def test_supported_preview_does_not_collapse_to_c2_while_wheel_is_behind(self):
    # Route 79 segment 39: the raw polynomial remains strongly outward, but
    # its instantaneous C3 demand falls below the maneuver-share threshold.
    desired_curvature = 0.0076
    controller = ProjectedLatControlPath()
    target = model(0.28, 0.09, 0.0074, 0.00083)

    command = controller.update(
      target, 0.0019, 10.3, True, False,
      projected_measured_curvature=0.0026,
      desired_angle_curvature=desired_curvature,
    )

    self.assertGreaterEqual(
      equivalent_curvature(command, 7.0),
      desired_curvature + 0.00025,
    )

  def test_supported_preview_extension_is_removed_at_projected_arrival(self):
    desired_curvature = 0.0076
    controller = ProjectedLatControlPath()
    target = model(0.28, 0.09, 0.0074, 0.00083)

    command = controller.update(
      target, 0.0019, 10.3, True, False,
      projected_measured_curvature=desired_curvature,
      desired_angle_curvature=desired_curvature,
    )

    self.assertLessEqual(equivalent_curvature(command, 7.0), desired_curvature)

  def test_supported_preview_retention_is_bounded_and_preserves_c2_c3(self):
    desired_curvature = 0.0076
    controller = ProjectedLatControlPath()
    target = model(0.28, 0.09, 0.0074, 0.00083)

    command = controller.update(
      target, 0.0019, 10.3, True, False,
      projected_measured_curvature=0.0026,
      desired_angle_curvature=desired_curvature,
    )

    extension = equivalent_curvature(command, 7.0) - desired_curvature
    self.assertGreater(extension, 0.0)
    self.assertLessEqual(extension, 0.0015 + 1e-12)
    self.assertEqual(command.curvature, target.curvature)
    self.assertEqual(command.curvature_rate, 0.0)

  def test_supported_preview_correction_is_bounded_across_large_shortfall(self):
    before = (0.0, 0.0, 0.0035, 0.0)
    after = _retain_feedback_preview(
      before,
      (0.05, 0.02, 0.0035, 0.0),
      0.02,
      0.001,
      0.001,
      1.0,
    )

    self.assertGreater(_curvature(after), _curvature(before))
    self.assertLessEqual(
      _curvature(after) - _curvature(before),
      FEEDBACK_PREVIEW_LIMIT + 1e-12,
    )

  def test_supported_preview_retention_waits_for_wheel_motion(self):
    desired_curvature = 0.0076
    controller = ProjectedLatControlPath()
    target = model(0.28, 0.09, 0.0074, 0.00083)

    command = controller.update(
      target, 0.0, 10.3, True, False,
      projected_measured_curvature=0.0,
      desired_angle_curvature=desired_curvature,
    )

    self.assertEqual(command.path_offset, 0.0)
    self.assertEqual(command.path_angle, 0.0)
    self.assertEqual(command.curvature, target.curvature)
    self.assertEqual(command.curvature_rate, 0.0)

  def test_supported_preview_retention_is_continuous_at_desired_maneuver_threshold(self):
    commands = []
    for desired_curvature in (0.002999, 0.003001):
      commands.append(ProjectedLatControlPath().update(
        model(0.28, 0.09, desired_curvature - 0.00025, 0.00083),
        0.0010,
        10.3,
        True,
        False,
        projected_measured_curvature=0.0012,
        desired_angle_curvature=desired_curvature,
      ))

    difference = equivalent_curvature(commands[1], 7.0) - \
      equivalent_curvature(commands[0], 7.0)
    self.assertGreaterEqual(difference, 0.0)
    self.assertLess(difference, 0.0001)

  def test_supported_preview_retention_is_continuous_at_local_support_threshold(self):
    desired_curvature = 0.0076
    commands = []
    for curvature_rate in (-0.00100001 / 7.0, -0.00099999 / 7.0):
      commands.append(ProjectedLatControlPath().update(
        model(0.28, 0.09, 0.001, curvature_rate),
        0.0006,
        10.3,
        True,
        False,
        projected_measured_curvature=0.0007,
        desired_angle_curvature=desired_curvature,
      ))

    difference = equivalent_curvature(commands[1], 7.0) - \
      equivalent_curvature(commands[0], 7.0)
    self.assertGreaterEqual(difference, 0.0)
    self.assertLess(difference, 0.0001)

  def test_supported_preview_retention_crossfades_into_polynomial(self):
    commands = []
    for curvature_rate in (0.0008737764, 0.0008737964):
      commands.append(ProjectedLatControlPath().update(
        model(0.28, 0.09, 0.0074, curvature_rate),
        0.0019,
        10.3,
        True,
        False,
        projected_measured_curvature=0.0026,
        desired_angle_curvature=0.0076,
      ))

    difference = equivalent_curvature(commands[1], 7.0) - \
      equivalent_curvature(commands[0], 7.0)
    self.assertLess(abs(difference), 0.0001)

  def test_supported_preview_retention_does_not_add_authority_at_pscm_limit(self):
    desired_curvature = 0.0076
    target = model(0.28, 0.09, 0.0074, 0.00083)

    for lat_ctl_limit in (1, 2, 3):
      with self.subTest(lat_ctl_limit=lat_ctl_limit):
        command = ProjectedLatControlPath().update(
          target, 0.0019, 10.3, True, False,
          projected_measured_curvature=0.0026,
          desired_angle_curvature=desired_curvature,
          lat_ctl_limit=lat_ctl_limit,
        )

        self.assertEqual(command.path_offset, 0.0)
        self.assertEqual(command.path_angle, 0.0)
        self.assertEqual(command.curvature, target.curvature)
        self.assertEqual(command.curvature_rate, 0.0)
