import unittest

from opendbc.car.ford.carcontroller import (
  get_ford_canfd_c0_lookahead,
  get_ford_canfd_c1_lookahead,
  get_ford_canfd_mode,
  get_ford_canfd_path_angle_bias,
  get_ford_curvature_filter_tau,
  shape_ford_canfd_curvature,
  suppress_curvature_sign_flip,
)
from opendbc.car.ford.values import CarControllerParams


class TestFordCanfdControllerHelpers(unittest.TestCase):
  def test_canfd_mode_prefers_extended_when_available(self):
    self.assertEqual(get_ford_canfd_mode(True, 2), 2)

  def test_canfd_mode_falls_back_to_limited(self):
    self.assertEqual(get_ford_canfd_mode(True, 1), 1)

  def test_canfd_mode_disables_without_capability(self):
    self.assertEqual(get_ford_canfd_mode(True, 0), 0)
    self.assertEqual(get_ford_canfd_mode(False, 2), 0)

  def test_curvature_sign_flip_is_suppressed_near_zero(self):
    self.assertEqual(suppress_curvature_sign_flip(-2e-4, 3e-4, 2e-4), 0.0)
    self.assertEqual(suppress_curvature_sign_flip(1e-4, 3e-4, 2e-4), 0.0)

  def test_curvature_sign_flip_keeps_meaningful_requests(self):
    self.assertEqual(suppress_curvature_sign_flip(-6e-4, 3e-4, 2e-4), -6e-4)

  def test_c0_lookahead_shrinks_with_speed(self):
    self.assertEqual(get_ford_canfd_c0_lookahead(5.0, 7.0), 7.0)
    self.assertAlmostEqual(get_ford_canfd_c0_lookahead(14.0, 14.0), 6.0)
    self.assertAlmostEqual(get_ford_canfd_c0_lookahead(25.0, 25.0), 4.0)

  def test_c1_centering_bias_builds_near_center(self):
    bias = get_ford_canfd_path_angle_bias(0.5, 0.01, 0.0005, 0.0, 15.0, True)
    self.assertGreater(bias, 0.0)
    self.assertLess(bias, 0.02)

  def test_c1_centering_bias_stays_off_in_turning(self):
    bias = get_ford_canfd_path_angle_bias(0.5, 0.08, 0.002, 0.01, 15.0, True)
    self.assertLess(bias, 0.01)

  def test_c1_lookahead_shrinks_on_exit(self):
    unwind_lookahead = get_ford_canfd_c1_lookahead(20.0, 20.0, 0.004, 0.002, 0)
    self.assertLess(unwind_lookahead, 20.0)

  def test_c1_lookahead_shrinks_near_limit(self):
    limited_lookahead = get_ford_canfd_c1_lookahead(20.0, 20.0, 0.002, 0.002, 2)
    self.assertLess(limited_lookahead, 20.0)
    self.assertGreaterEqual(limited_lookahead, CarControllerParams.C1_LOOKAHEAD_MIN)

  def test_curvature_filter_tau_is_asymmetric(self):
    self.assertEqual(get_ford_curvature_filter_tau(0.004, 0.002), CarControllerParams.C2_WINDUP_TAU)
    self.assertEqual(get_ford_curvature_filter_tau(0.001, 0.002), CarControllerParams.C2_UNWIND_TAU)
    self.assertEqual(get_ford_curvature_filter_tau(-0.001, 0.002), CarControllerParams.C2_CROSSOVER_TAU)

  def test_curvature_windup_reduces_near_limit(self):
    limited = shape_ford_canfd_curvature(0.004, 0.002, 2, 2)
    unrestricted = shape_ford_canfd_curvature(0.004, 0.002, 0, 2)
    self.assertLess(limited, unrestricted)

  def test_curvature_unwind_is_not_scaled(self):
    self.assertEqual(shape_ford_canfd_curvature(0.001, 0.002, 2, 1), 0.001)

if __name__ == "__main__":
  unittest.main()
