from types import SimpleNamespace
import math
import unittest

from opendbc.car.ford.lateral_path_servo import FordPolynomialServo, Lmc2Polynomial, SteeringFeedback


def model(path_offset: float, path_angle: float, curvature: float = 0.0, curvature_rate: float = 0.0):
  return SimpleNamespace(
    valid=True,
    pathOffset=path_offset,
    pathAngle=path_angle,
    curvature=curvature,
    curvatureRate=curvature_rate,
  )


def feedback(
  desired: float,
  measured: float,
  projected: float,
  *,
  desired_angle: float | None = None,
  speed: float = 7.0,
  active: bool = True,
  driver_override: bool = False,
  lat_ctl_limit: int = 0,
):
  return SteeringFeedback(
    measured_curvature=measured,
    projected_curvature=projected,
    desired_angle_curvature=desired,
    desired_angle_deg=desired * 4000.0 if desired_angle is None else desired_angle,
    speed=speed,
    active=active,
    driver_override=driver_override,
    lat_ctl_limit=lat_ctl_limit,
  )


def equivalent_curvature(command, distance: float = 7.0) -> float:
  path_offset = getattr(command, "path_offset", getattr(command, "pathOffset", 0.0))
  path_angle = getattr(command, "path_angle", getattr(command, "pathAngle", 0.0))
  curvature_rate = getattr(command, "curvature_rate", getattr(command, "curvatureRate", 0.0))
  return 2.0 * path_offset / distance**2 + 2.0 * path_angle / distance + command.curvature + curvature_rate * distance / 3.0


class TestFordPolynomialServo(unittest.TestCase):
  def test_ordinary_steady_driving_is_c2_only(self):
    servo = FordPolynomialServo()
    curvature = 0.015
    path = model(
      0.5 * curvature * 7.0**2,
      curvature * 7.0,
      curvature,
    )

    command = servo.update(path, feedback(curvature, curvature, curvature))

    assert command.path_offset == 0.0
    assert command.path_angle == 0.0
    assert command.curvature == curvature
    assert command.curvature_rate == 0.0

  def test_full_polynomial_authority_is_available_while_wheel_is_behind(self):
    servo = FordPolynomialServo()
    path = model(2.885494, 0.558042, 0.00015, 0.010035)

    command = servo.update(path, feedback(0.07, 0.0, 0.0, speed=3.7))

    assert equivalent_curvature(command) > 0.25
    assert command.path_offset >= path.pathOffset
    assert command.path_angle == 0.497
    assert command.curvature == 0.0
    assert command.curvature_rate == 0.001023

  def test_retreating_desired_caps_outward_command_after_wheel_passes_target(self):
    servo = FordPolynomialServo()
    path = model(2.885494, 0.558042, 0.00015, 0.010035)
    desired_profile = (
      (0.04, 160.0),
      (0.06, 240.0),
      (0.07, 280.0),
      (0.06, 240.0),
      (0.04, 160.0),
      (0.02, 80.0),
    )

    command = None
    for desired, desired_angle in desired_profile:
      command = servo.update(
        path,
        feedback(desired, 0.08, 0.09, desired_angle=desired_angle, speed=3.7),
      )

    assert command is not None
    assert 0.0 < equivalent_curvature(command) <= 0.026001

  def test_terminal_envelope_remains_active_after_desired_settles(self):
    servo = FordPolynomialServo()
    path = model(2.885494, 0.558042, 0.00015, 0.010035)
    for desired, desired_angle in (
      (0.04, 160.0),
      (0.06, 240.0),
      (0.07, 280.0),
      (0.06, 240.0),
      (0.04, 160.0),
      (0.02, 80.0),
    ):
      servo.update(path, feedback(desired, 0.08, 0.09, desired_angle=desired_angle, speed=3.7))

    command = None
    for _ in range(10):
      command = servo.update(path, feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7))

    assert command is not None
    assert 0.0 < equivalent_curvature(command) <= 0.026001

  def test_new_outward_target_relatches_full_authority_in_same_update(self):
    servo = FordPolynomialServo()
    path = model(2.885494, 0.558042, 0.00015, 0.010035)
    for desired, desired_angle in (
      (0.04, 160.0),
      (0.06, 240.0),
      (0.07, 280.0),
      (0.06, 240.0),
      (0.04, 160.0),
      (0.02, 80.0),
    ):
      servo.update(path, feedback(desired, 0.08, 0.09, desired_angle=desired_angle, speed=3.7))

    command = servo.update(path, feedback(0.10, 0.08, 0.09, desired_angle=400.0, speed=3.7))

    assert equivalent_curvature(command) > 0.25

  def test_terminal_envelope_never_suppresses_model_unwind_geometry(self):
    servo = FordPolynomialServo()
    outward_path = model(2.885494, 0.558042, 0.00015, 0.010035)
    for desired, desired_angle in (
      (0.04, 160.0),
      (0.06, 240.0),
      (0.07, 280.0),
      (0.06, 240.0),
      (0.04, 160.0),
      (0.02, 80.0),
    ):
      servo.update(outward_path, feedback(desired, 0.08, 0.09, desired_angle=desired_angle, speed=3.7))

    unwind_path = model(2.0, 0.3, 0.00015, -0.001)
    command = servo.update(unwind_path, feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7))

    assert command.curvature_rate == -0.001
    assert 0.0 < equivalent_curvature(command) <= 0.026001
    assert command.path_offset < unwind_path.pathOffset
    assert command.path_angle < unwind_path.pathAngle

  def test_model_unwind_starts_terminal_protection_without_waiting_for_angle_history(self):
    servo = FordPolynomialServo()
    path = model(2.0, 0.3, 0.00015, -0.001)

    command = servo.update(
      path,
      feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7),
    )

    assert command.curvature_rate == -0.001
    assert 0.0 < equivalent_curvature(command) <= 0.026001

  def test_driver_override_commands_the_measured_wheel_path(self):
    servo = FordPolynomialServo()
    path = model(2.0, 0.3, 0.01, 0.001)

    command = servo.update(
      path,
      feedback(0.03, 0.02, 0.025, speed=10.0, driver_override=True),
    )

    assert command.path_offset == 0.5 * 0.02 * 7.0**2
    assert command.path_angle == 0.02 * 10.0
    assert command.curvature == 0.0
    assert command.curvature_rate == 0.0

  def test_c2_baseband_never_reverses_meaningful_model_geometry(self):
    servo = FordPolynomialServo()
    path = model(
      -0.1324408266,
      -0.0564401015,
      0.0084251088,
      -0.0005080627,
    )

    command = servo.update(path, feedback(-0.005, -0.005, -0.005, speed=10.0))

    assert equivalent_curvature(path) < -0.01
    assert equivalent_curvature(command) <= 0.0

  def test_conflicting_c3_cannot_cancel_a_turn_while_wheel_is_behind(self):
    servo = FordPolynomialServo()
    path = model(-0.000848, 0.009214, -0.021576, 0.0045)

    command = servo.update(
      path,
      feedback(-0.021576, -0.0101, -0.0101, speed=4.67),
    )

    assert equivalent_curvature(command) < -0.05
    assert command.curvature_rate == 0.0

  def test_undertracking_extension_is_one_sided_and_immediately_removable(self):
    for direction in (-1.0, 1.0):
      servo = FordPolynomialServo()
      path = model(
        direction * 0.5 * 0.002 * 7.0**2,
        direction * 0.002 * 7.0,
        direction * 0.002,
      )

      behind = servo.update(
        path,
        feedback(direction * 0.004, 0.0, 0.0),
      )
      arrived = servo.update(
        path,
        feedback(direction * 0.004, direction * 0.004, direction * 0.004),
      )

      assert 0.002 < direction * equivalent_curvature(behind) <= 0.004
      assert direction * equivalent_curvature(arrived) == 0.002

  def test_invalid_path_falls_back_to_bounded_c2_without_losing_authority(self):
    servo = FordPolynomialServo()
    for direction in (-1.0, 1.0):
      path = SimpleNamespace(
        valid=False,
        pathOffset=2.0,
        pathAngle=0.3,
        curvature=direction * 0.05,
        curvatureRate=0.001,
      )

      command = servo.update(path, feedback(direction * 0.05, 0.0, 0.0))

      assert command.valid is False
      assert command.path_offset == 0.0
      assert command.path_angle == 0.0
      assert command.curvature == direction * 0.02
      assert command.curvature_rate == 0.0

  def test_invalid_path_clears_terminal_history(self):
    servo = FordPolynomialServo()
    servo.update(
      model(2.0, 0.3, 0.00015, -0.001),
      feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7),
    )
    servo.update(
      SimpleNamespace(valid=False, curvature=0.0),
      feedback(0.0, 0.0, 0.0),
    )

    command = servo.update(
      model(2.885494, 0.558042, 0.00015),
      feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7),
    )

    assert equivalent_curvature(command) > 0.20

  def test_subthreshold_c3_noise_does_not_latch_terminal_protection(self):
    servo = FordPolynomialServo()
    servo.update(
      model(0.245, 0.07, 0.01, -0.0001),
      feedback(0.01, 0.02, 0.02, desired_angle=40.0),
    )

    command = servo.update(
      model(2.885494, 0.558042, 0.00015),
      feedback(0.02, 0.08, 0.09, desired_angle=80.0, speed=3.7),
    )

    assert equivalent_curvature(command) > 0.20

  def test_fresh_pinned_outward_c3_is_guarded_after_wheel_arrival(self):
    for direction in (-1.0, 1.0):
      command = FordPolynomialServo().update(
        model(
          direction * 2.885494,
          direction * 0.558042,
          direction * 0.00015,
          direction * 0.010035,
        ),
        feedback(
          direction * 0.00015,
          direction * 0.05,
          direction * 0.06,
          desired_angle=direction * 0.6,
          speed=3.7,
          lat_ctl_limit=2,
        ),
      )

      assert 0.0 < direction * equivalent_curvature(command) <= 0.006151

  def test_inactive_or_missing_path_returns_zero(self):
    servo = FordPolynomialServo()

    assert servo.update(None, feedback(0.0, 0.0, 0.0)) == Lmc2Polynomial()
    assert (
      servo.update(
        model(1.0, 0.2, 0.01, 0.001),
        feedback(0.01, 0.0, 0.0, active=False),
      )
      == Lmc2Polynomial()
    )

  def test_pscm_limit_state_moves_outward_c3_nearer_without_changing_15m_path(self):
    path = model(1.6, 0.35, 0.03, 0.005)
    unrestricted = FordPolynomialServo().update(
      path,
      feedback(0.04, 0.01, 0.01),
    )
    limit_close = FordPolynomialServo().update(
      path,
      feedback(0.04, 0.01, 0.01, lat_ctl_limit=1),
    )

    assert limit_close.path_offset > unrestricted.path_offset
    assert 0.0 < limit_close.curvature_rate < unrestricted.curvature_rate
    assert abs(equivalent_curvature(limit_close, 15.0) - equivalent_curvature(unrestricted, 15.0)) < 1e-12

  def test_subthreshold_model_noise_keeps_ordinary_driving_on_c2(self):
    servo = FordPolynomialServo()
    path = model(
      0.5 * 0.012 * 7.0**2,
      0.008 * 7.0,
      0.01,
      0.0001,
    )

    command = servo.update(path, feedback(0.01, 0.011, 0.009))

    assert command == Lmc2Polynomial(valid=True, curvature=0.01)

  def test_interior_commands_are_left_right_symmetric(self):
    positive_path = model(0.3, 0.05, 0.005, 0.001)
    negative_path = model(-0.3, -0.05, -0.005, -0.001)

    positive = FordPolynomialServo().update(
      positive_path,
      feedback(0.02, 0.0, 0.0, desired_angle=80.0),
    )
    negative = FordPolynomialServo().update(
      negative_path,
      feedback(-0.02, 0.0, 0.0, desired_angle=-80.0),
    )

    positive_values = (
      positive.path_offset,
      positive.path_angle,
      positive.curvature,
      positive.curvature_rate,
    )
    negative_values = (
      negative.path_offset,
      negative.path_angle,
      negative.curvature,
      negative.curvature_rate,
    )
    assert all(abs(left + right) < 1e-12 for left, right in zip(positive_values, negative_values, strict=True))

  def test_every_coefficient_is_finite_and_inside_asymmetric_dbc_bounds(self):
    limits = (
      (-4.61, 4.60),
      (-0.475, 0.497),
      (-0.02, 0.02),
      (-0.001024, 0.001023),
    )
    cases = (
      model(100.0, 100.0, 100.0, 100.0),
      model(-100.0, -100.0, -100.0, -100.0),
      model(math.nan, math.inf, -math.inf, math.nan),
    )

    for path in cases:
      command = FordPolynomialServo().update(
        path,
        feedback(0.1, 0.0, 0.0, lat_ctl_limit=2),
      )
      values = (
        command.path_offset,
        command.path_angle,
        command.curvature,
        command.curvature_rate,
      )
      assert all(math.isfinite(value) for value in values)
      assert all(lower <= value <= upper for value, (lower, upper) in zip(values, limits, strict=True))
