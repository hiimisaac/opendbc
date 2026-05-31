#!/usr/bin/env python3
"""Anticipator (phase-lead) for the Ford bal controller.

Ported from the proven sp-dev-c3 angle controller, adapted to the deep_rl
port's 20 Hz call rate (STEER_STEP=5 -> 50 ms/frame, vs 100 Hz on sp-dev-c3).

Purpose: feed bal_encode `desk + tau * d(desk)/dt` instead of raw `desk`, so
the command leads the PSCM's ~1 s highway lag. This adds phase margin and
breaks the ~0.6 Hz closed-loop limit cycle (the "going wide then pulling back
hard" weave on 45 mph curves) WITHOUT lowering the steady gain (no wide).
"""
from opendbc.car.ford.lateral_bal import FordAnticipator, FORD_BAL_ANTICIPATOR_TAU, FORD_BAL_DT


def _fill(ant, desk, n):
  """Push `desk` n times; return the last output."""
  out = desk
  for _ in range(n):
    out = ant.update(desk, True)
  return out


def test_steady_desk_returns_desk_unchanged():
  """No lead on a held curve: d(desk)/dt = 0 -> anticipated == desk."""
  ant = FordAnticipator()
  out = _fill(ant, 0.012, 10)
  assert abs(out - 0.012) < 1e-9


def test_before_window_fills_returns_raw_desk():
  """First frame(s) post-engage must not spike: pass desk through until the
  derivative window has enough samples."""
  ant = FordAnticipator()
  assert ant.update(0.02, True) == 0.02  # only one sample


def test_rising_desk_leads_positive():
  """Entering a turn (desk ramping up): anticipated > desk (pre-loads)."""
  ant = FordAnticipator()
  out = 0.0
  for i in range(1, ant.window + 1):
    out = ant.update(0.002 * i, True)  # +0.002 / frame
  # dot = +0.002/FORD_BAL_DT ; lead = TAU*dot > 0
  assert out > 0.002 * ant.window
  expected_dot = 0.002 / FORD_BAL_DT
  assert abs(out - (0.002 * ant.window + FORD_BAL_ANTICIPATOR_TAU * expected_dot)) < 1e-6


def test_falling_desk_leads_negative():
  """Exiting a turn (desk ramping down toward zero from a positive hold):
  anticipated < desk so commands fall faster than the planner."""
  ant = FordAnticipator()
  # start held high, then ramp down but stay positive
  _fill(ant, 0.05, ant.window)
  out = ant.update(0.045, True)
  assert out < 0.045


def test_sign_flip_guard_caps_at_zero():
  """A rapid drop whose lead would swing past zero into the opposite turn is
  capped at 0 -- anticipation alone never commands an opposite-direction curve."""
  ant = FordAnticipator()
  _fill(ant, 0.02, ant.window)            # held positive
  out = ant.update(0.0005, True)          # sudden near-zero -> big negative dot
  assert out >= 0.0                       # never flips negative
  # mirror: held negative, sudden near-zero -> never flips positive
  ant2 = FordAnticipator()
  _fill(ant2, -0.02, ant2.window)
  out2 = ant2.update(-0.0005, True)
  assert out2 <= 0.0


def test_disengage_resets_history():
  """Not lat_active clears state so the first re-engaged frame can't spike off
  a stale pre-disengage derivative."""
  ant = FordAnticipator()
  _fill(ant, 0.05, ant.window)
  assert ant.update(0.05, False) == 0.05  # passthrough + reset
  # next active frame is treated as a fresh start (window not yet full)
  assert ant.update(0.0, True) == 0.0
