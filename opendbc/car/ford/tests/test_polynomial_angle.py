import math

from opendbc.car.ford.polynomial_angle import (
  FORD_POLY_LOOKAHEAD,
  polynomial_angle_command,
)


def step(desired, angle_error, measured, *, virtual_last=0.0, path_last=0.0, rate_last=0.0,
         v_ego=7.0, driver_override=False, driver_handoff=False):
  return polynomial_angle_command(desired, angle_error, measured, v_ego, True, driver_override,
                                  virtual_curvature_last=virtual_last,
                                  path_curvature_last=path_last,
                                  curvature_rate_last=rate_last,
                                  driver_handoff=driver_handoff)


def test_polynomial_terms_are_one_virtual_path_command():
  cmd = step(0.03, 0.004, 0.0)

  # C0 and C1 are two geometric encodings of one path-curvature command,
  # rather than independently tuned steering channels.
  assert math.isclose(cmd.path_angle, cmd.path_curvature * FORD_POLY_LOOKAHEAD)
  assert math.isclose(cmd.path_offset, 0.5 * cmd.path_curvature * FORD_POLY_LOOKAHEAD ** 2)
  assert cmd.curvature > 0.0
  assert cmd.curvature_rate > 0.0


def test_falling_angle_target_cannot_reintroduce_old_direction_c2():
  virtual = path = rate = 0.0
  c2_values = []
  for desired in (0.012, 0.0096, 0.0072, 0.0048, 0.0024, 0.0):
    # The wheel is still carrying +0.010/m of the old turn while the requested
    # steering angle is returning to straight.
    angle_error = desired - 0.010
    cmd = step(desired, angle_error, 0.010, virtual_last=virtual, path_last=path, rate_last=rate)
    virtual, path, rate = cmd.virtual_curvature, cmd.path_curvature, cmd.curvature_rate
    c2_values.append(cmd.curvature)

  assert all(b <= a for a, b in zip(c2_values, c2_values[1:], strict=False))
  assert cmd.curvature == 0.0
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0
  assert cmd.curvature_rate <= 0.0


def test_release_and_reversal_are_not_delayed_by_attack_limit():
  released = step(0.0, -0.01, 0.01, virtual_last=0.03, path_last=0.03, rate_last=0.001)
  assert released.virtual_curvature == 0.0
  assert released.path_curvature < 0.0
  assert released.curvature_rate < 0.0

  reversed_cmd = step(-0.02, -0.02, 0.01, virtual_last=0.03, path_last=0.03, rate_last=0.001)
  assert reversed_cmd.virtual_curvature < 0.0
  assert reversed_cmd.path_curvature < 0.0
  assert reversed_cmd.curvature_rate < 0.0


def test_driver_override_tracks_the_held_arc_as_one_polynomial():
  cmd = step(0.03, 0.02, -0.015, driver_override=True)

  assert cmd.virtual_curvature == -0.015
  assert cmd.curvature < 0.0
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0
  assert cmd.curvature_rate == 0.0


def test_inactive_command_clears_all_polynomial_state():
  cmd = polynomial_angle_command(0.03, 0.02, 0.01, 7.0, False, False,
                                 virtual_curvature_last=0.02,
                                 path_curvature_last=0.02,
                                 curvature_rate_last=0.001)

  assert cmd.curvature == 0.0
  assert cmd.curvature_rate == 0.0
  assert cmd.path_angle == 0.0
  assert cmd.path_offset == 0.0
  assert cmd.virtual_curvature == 0.0
  assert cmd.path_curvature == 0.0
  assert cmd.handoff_complete
