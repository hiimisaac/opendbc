import math

from opendbc.car.ford.polynomial_angle import (
  FORD_POLY_LOOKAHEAD,
  polynomial_angle_command,
)


def step(desired, angle_error, measured, *, virtual_last=0.0, path_last=0.0, rate_last=0.0,
         v_ego=7.0, driver_override=False, driver_handoff=False,
         c2_latched_last=False, c2_recovery_frames_last=0):
  return polynomial_angle_command(desired, angle_error, measured, v_ego, True, driver_override,
                                  virtual_curvature_last=virtual_last,
                                  path_curvature_last=path_last,
                                  curvature_rate_last=rate_last,
                                  driver_handoff=driver_handoff,
                                  c2_latched_last=c2_latched_last,
                                  c2_recovery_frames_last=c2_recovery_frames_last)


def test_polynomial_terms_are_one_virtual_path_command():
  cmd = step(0.01, 0.004, 0.0)

  # C0 and C1 are two geometric encodings of one path-curvature command,
  # rather than independently tuned steering channels.
  assert math.isclose(cmd.path_angle, cmd.path_curvature * FORD_POLY_LOOKAHEAD)
  assert math.isclose(cmd.path_offset, 0.5 * cmd.path_curvature * FORD_POLY_LOOKAHEAD ** 2)
  assert cmd.curvature > 0.0
  assert cmd.curvature_rate > 0.0


def test_falling_angle_target_cannot_reintroduce_old_direction_c2():
  virtual = path = rate = 0.0
  latched = False
  recovery = 0
  c2_values = []
  for desired in (0.012, 0.0096, 0.0072, 0.0048, 0.0024, 0.0):
    # The wheel is still carrying +0.010/m of the old turn while the requested
    # steering angle is returning to straight.
    angle_error = desired - 0.010
    cmd = step(desired, angle_error, 0.010, virtual_last=virtual, path_last=path, rate_last=rate,
               c2_latched_last=latched, c2_recovery_frames_last=recovery)
    virtual, path, rate = cmd.virtual_curvature, cmd.path_curvature, cmd.curvature_rate
    latched, recovery = cmd.c2_latched, cmd.c2_recovery_frames
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
  assert cmd.curvature == 0.0
  assert cmd.path_angle < 0.0
  assert cmd.path_offset < 0.0
  assert cmd.curvature_rate == 0.0


def test_large_turn_latches_c2_at_zero_through_exit():
  virtual = path = rate = 0.0
  latched = False
  recovery = 0
  commands = []
  for desired, angle_error in ((0.013, 0.013), (0.020, 0.012), (0.009, 0.006),
                               (0.004, -0.003), (0.0, -0.010)):
    cmd = step(desired, angle_error, 0.010, virtual_last=virtual, path_last=path, rate_last=rate,
               c2_latched_last=latched, c2_recovery_frames_last=recovery)
    virtual, path, rate = cmd.virtual_curvature, cmd.path_curvature, cmd.curvature_rate
    latched, recovery = cmd.c2_latched, cmd.c2_recovery_frames
    commands.append(cmd)

  assert all(cmd.curvature == 0.0 for cmd in commands)
  assert commands[-1].c2_latched
  assert commands[-1].path_curvature < 0.0


def test_c2_returns_only_after_near_straight_flush_interval():
  virtual = path = rate = 0.0
  latched = True
  recovery = 0

  # A gentle same-direction request is not enough to re-enable C2 while the
  # steering-angle error still says the truck is carrying the old turn.
  cmd = step(0.004, -0.004, 0.008, c2_latched_last=latched)
  assert cmd.curvature == 0.0
  assert cmd.c2_latched

  # Both target and angle error must remain near straight for the full flush.
  for _ in range(10):
    cmd = step(0.001, 0.0005, 0.001, virtual_last=virtual, path_last=path, rate_last=rate,
               c2_latched_last=latched, c2_recovery_frames_last=recovery)
    virtual, path, rate = cmd.virtual_curvature, cmd.path_curvature, cmd.curvature_rate
    latched, recovery = cmd.c2_latched, cmd.c2_recovery_frames

  assert not cmd.c2_latched
  assert cmd.curvature == 0.001


def test_inactive_frames_finish_c2_flush_before_reenable():
  latched = True
  recovery = 0
  for _ in range(9):
    cmd = polynomial_angle_command(0.0, 0.0, 0.0, 7.0, False, False,
                                   c2_latched_last=latched,
                                   c2_recovery_frames_last=recovery)
    latched, recovery = cmd.c2_latched, cmd.c2_recovery_frames

  assert cmd.c2_latched

  cmd = polynomial_angle_command(0.0, 0.0, 0.0, 7.0, False, False,
                                 c2_latched_last=latched,
                                 c2_recovery_frames_last=recovery)
  assert not cmd.c2_latched


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
  assert not cmd.c2_latched
