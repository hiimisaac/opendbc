import math
from types import SimpleNamespace

from opendbc.car.ford.coherent_path import (
  FORD_COHERENT_C0_LIMITS,
  FORD_COHERENT_C1_LIMITS,
  FORD_COHERENT_C3_LIMITS,
  CoherentPathController,
)


def arc_model(curvature, distances=(0.0, 1.75, 3.5, 5.0, 7.0)):
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * curvature * distance * distance for distance in distances],
    ),
    orientation=SimpleNamespace(z=[curvature * distance for distance in distances]),
  )


def effective_path_offset(command, distance):
  return command.path_offset + command.path_angle * distance + \
         0.5 * command.estimated_c2 * distance * distance + \
         command.curvature_rate * distance ** 3 / 6.0


def effective_path_heading(command, distance):
  return command.path_angle + command.estimated_c2 * distance + 0.5 * command.curvature_rate * distance ** 2


def polynomial_model(c0, c1, c2, c3, distances=(0.0, 1.75, 3.5, 5.0, 7.0)):
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[c0 + c1 * s + 0.5 * c2 * s ** 2 + c3 * s ** 3 / 6.0 for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[c1 + c2 * s + 0.5 * c3 * s ** 2 for s in distances],
    ),
  )


def test_retained_c2_is_subtracted_from_straight_release_path():
  controller = CoherentPathController()
  for _ in range(100):
    controller.update(arc_model(0.003), 0.003, 0.003, 10.0, True, False, 0.0)

  command = controller.update(arc_model(0.0), 0.0, 0.003, 10.0, True, False, 0.0)

  assert command.curvature == 0.0
  assert command.estimated_c2 > 0.0025
  assert math.isclose(effective_path_offset(command, 7.0), 0.0, abs_tol=0.03)
  assert command.path_angle < 0.0 or command.path_offset < 0.0 or command.curvature_rate < 0.0


def test_first_c2_command_does_not_advance_effective_state_early():
  controller = CoherentPathController()

  first = controller.update(arc_model(0.003), 0.003, 0.0, 10.0, True, False, 0.0)
  second = controller.update(arc_model(0.003), 0.003, 0.0, 10.0, True, False, 0.0)

  assert first.curvature == 0.0002
  assert first.estimated_c2 == 0.0
  assert second.estimated_c2 > 0.0


def test_release_and_reversal_first_finish_previous_c2_interval():
  controllers = (CoherentPathController(), CoherentPathController())
  for controller in controllers:
    for _ in range(20):
      controller.update(arc_model(0.003), 0.003, 0.0, 10.0, True, False, 0.0)

  release_before = controllers[0].estimated_c2
  reversal_before = controllers[1].estimated_c2
  release = controllers[0].update(arc_model(0.0), 0.0, 0.0, 10.0, True, False, 0.0)
  reversal = controllers[1].update(arc_model(-0.003), -0.003, 0.0, 10.0, True, False, 0.0)

  assert release.curvature == 0.0
  assert reversal.curvature == 0.0
  assert release.estimated_c2 > release_before
  assert reversal.estimated_c2 > reversal_before


def test_retained_c2_compensation_is_sign_symmetric():
  commands = []
  for sign in (-1.0, 1.0):
    controller = CoherentPathController()
    for _ in range(100):
      controller.update(arc_model(sign * 0.003), sign * 0.003, sign * 0.003, 10.0, True, False, 0.0)
    commands.append(controller.update(arc_model(0.0), 0.0, sign * 0.003, 10.0, True, False, 0.0))

  negative, positive = commands
  assert math.isclose(negative.estimated_c2, -positive.estimated_c2, abs_tol=1e-12)
  assert math.isclose(negative.path_offset, -positive.path_offset, abs_tol=1e-12)
  assert math.isclose(negative.path_angle, -positive.path_angle, abs_tol=1e-12)
  assert math.isclose(negative.curvature_rate, -positive.curvature_rate, abs_tol=1e-12)


def test_retained_c2_estimate_and_compensation_decay_continuously():
  controller = CoherentPathController()
  for _ in range(100):
    controller.update(arc_model(0.003), 0.003, 0.003, 10.0, True, False, 0.0)

  commands = [controller.update(arc_model(0.0), 0.0, 0.003, 10.0, True, False, 0.0) for _ in range(20)]

  assert all(current.estimated_c2 < previous.estimated_c2 for previous, current in zip(commands, commands[1:], strict=False))
  assert abs(commands[-1].path_angle) < abs(commands[0].path_angle)
  assert abs(commands[-1].curvature_rate) < abs(commands[0].curvature_rate)
  assert all(math.isclose(effective_path_offset(command, 7.0), 0.0, abs_tol=0.03) for command in commands)


def test_c2_reversal_flushes_wire_sign_but_effective_path_reverses_immediately():
  controller = CoherentPathController()
  for _ in range(100):
    controller.update(arc_model(0.003), 0.003, 0.003, 10.0, True, False, 0.0)

  command = controller.update(arc_model(-0.003), -0.003, 0.003, 10.0, True, False, 0.0)

  assert command.curvature == 0.0
  assert command.estimated_c2 > 0.0
  assert effective_path_offset(command, 7.0) < 0.0
  assert math.isclose(effective_path_offset(command, 7.0), -0.5 * 0.003 * 7.0 ** 2, abs_tol=0.03)


def test_model_action_and_spatial_shape_are_projected_as_one_polynomial():
  controller = CoherentPathController()
  model = polynomial_model(0.08, -0.015, 0.004, 0.00035)

  command = controller.update(model, 0.004, 0.004, 10.0, True, False, 0.0)

  for distance in (0.0, 1.75, 3.5, 5.0, 7.0):
    expected_offset = 0.08 - 0.015 * distance + 0.5 * 0.004 * distance ** 2 + 0.00035 * distance ** 3 / 6.0
    expected_heading = -0.015 + 0.004 * distance + 0.5 * 0.00035 * distance ** 2
    assert math.isclose(effective_path_offset(command, distance), expected_offset, abs_tol=0.01)
    # At initial engagement the effective C2 estimate is still near zero, so
    # the bounded C3 channel cannot reproduce a quadratic exactly. Position
    # remains tight and the joint least-squares projection bounds heading error.
    assert math.isclose(effective_path_heading(command, distance), expected_heading, abs_tol=0.015)


def test_model_path_uses_spatial_distance_when_ego_x_reverses():
  controller = CoherentPathController()
  model = SimpleNamespace(
    position=SimpleNamespace(
      x=[0.0, 3.0, 2.0, 1.0, 0.0],
      y=[0.0, 0.1, 0.3, 0.6, 1.0],
    ),
    orientation=SimpleNamespace(z=[0.0, 0.04, 0.12, 0.25, 0.4]),
  )

  command = controller.update(model, 0.0, 0.0, 5.0, True, False, 0.0)

  # A longitudinal-x sampler rejects this valid sharp path and falls back to a
  # zero arc. Spatial sampling must preserve its lateral model geometry.
  assert effective_path_offset(command, 5.0) > 0.2
  assert effective_path_heading(command, 5.0) > 0.05


def test_impossible_path_stays_inside_every_wire_bound():
  controller = CoherentPathController()
  command = controller.update(arc_model(0.2), 0.2, 0.0, 10.0, True, False, 0.02)

  assert FORD_COHERENT_C0_LIMITS[0] <= command.path_offset <= FORD_COHERENT_C0_LIMITS[1]
  assert FORD_COHERENT_C1_LIMITS[0] <= command.path_angle <= FORD_COHERENT_C1_LIMITS[1]
  assert FORD_COHERENT_C3_LIMITS[0] <= command.curvature_rate <= FORD_COHERENT_C3_LIMITS[1]


def test_driver_override_tracks_held_wheel_arc_without_charging_c2():
  controller = CoherentPathController()
  command = controller.update(arc_model(-0.01), -0.01, 0.012, 5.0, True, True, 0.0)

  assert command.curvature == 0.0
  assert command.cooperative_control
  assert math.isclose(effective_path_offset(command, 7.0), 0.5 * 0.012 * 7.0 ** 2, abs_tol=0.03)


def test_driver_handoff_completes_while_model_remains_in_a_curve():
  controller = CoherentPathController()
  controller.update(arc_model(0.02), 0.02, -0.02, 5.0, True, True, 0.0)

  commands = [controller.update(arc_model(-0.015), -0.015, -0.015, 5.0, True, False, 0.0) for _ in range(20)]

  assert commands[0].cooperative_control
  assert not commands[-1].cooperative_control
  assert effective_path_offset(commands[-1], 7.0) < 0.0


def test_inactive_output_is_zero_while_retained_c2_estimate_decays():
  controller = CoherentPathController()
  for _ in range(100):
    controller.update(arc_model(0.003), 0.003, 0.003, 10.0, True, False, 0.0)
  before = controller.estimated_c2

  command = controller.update(None, 0.0, 0.0, 0.0, False, False, 0.0)
  next_command = controller.update(None, 0.0, 0.0, 0.0, False, False, 0.0)

  assert (command.curvature, command.curvature_rate, command.path_angle, command.path_offset) == (0.0, 0.0, 0.0, 0.0)
  assert command.estimated_c2 > before
  assert 0.0 < next_command.estimated_c2 < command.estimated_c2


def test_missing_and_nonfinite_inputs_fall_back_to_a_finite_arc():
  controller = CoherentPathController()
  command = controller.update(None, float("nan"), float("inf"), float("nan"), True, False, float("nan"))

  assert all(math.isfinite(value) for value in (
    command.curvature, command.curvature_rate, command.path_angle,
    command.path_offset, command.estimated_c2,
  ))
