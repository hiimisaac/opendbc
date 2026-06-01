#!/usr/bin/env python3
"""Cut the undamped c0 (offset) channel above ~10 m/s.

The sustained-curve bounce is a ~0.5 Hz resonance where realized act swings ~3x
more than c1 can deliver through the (attenuating) plant — an amplification c1
alone can't produce. The one undamped element we feed at speed is c0 (~0.3 m on
curves, because g0 is tiny). c0 is the known hunting channel; c1 alone drives
curves cleanly. So we remove c0 above ~10 m/s and let c1 carry the curve.
"""
from opendbc.car.ford.lateral_bal import bal_encode, FORD_WBAL_C0_VMAX


def test_c0_off_at_highway_speed():
  """Above FORD_WBAL_C0_VMAX, c0 must be zero on a curve (c1 carries it)."""
  c0, c1, _ = bal_encode(0.012, 14.0, None, 1.0)
  assert c0 == 0.0
  assert abs(c1) > 0.0


def test_c0_active_at_low_speed():
  """Below the ramp, c0 still provides low-speed offset on a real curve."""
  c0, c1, _ = bal_encode(0.03, 5.0, None, 1.0)
  assert abs(c0) > 0.0


def test_c0_fades_not_steps_across_ramp():
  """c0 share tapers to 0 across the speed ramp (no discontinuity)."""
  desk = 0.02
  c0_lo, _, _ = bal_encode(desk, 8.0, None, 1.0)   # ramp start: still some c0
  c0_mid, _, _ = bal_encode(desk, 9.0, None, 1.0)  # mid-ramp: less
  c0_hi, _, _ = bal_encode(desk, 10.0, None, 1.0)  # ramp end: zero
  assert abs(c0_hi) <= abs(c0_mid) <= abs(c0_lo)
  assert c0_hi == 0.0


def test_delivery_preserved_when_c0_cut():
  """Cutting c0 must not drop delivery — c1 picks up the full curve (g1·c1 ≈ desk)."""
  from opendbc.car.ford.lateral_bal import _interp, FORD_WBAL_GC1_V, FORD_WBAL_GC1, FORD_WBAL_GC1_TRIM_V, FORD_WBAL_GC1_TRIM
  desk, v = 0.012, 14.0
  c0, c1, _ = bal_encode(desk, v, None, 1.0)
  g1 = _interp(v, FORD_WBAL_GC1_V, FORD_WBAL_GC1) * _interp(v, FORD_WBAL_GC1_TRIM_V, FORD_WBAL_GC1_TRIM)
  assert c0 == 0.0
  assert abs(g1 * c1 - desk) < 1e-3   # c1 alone delivers the full curve
