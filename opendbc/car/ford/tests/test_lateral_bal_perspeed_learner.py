#!/usr/bin/env python3
"""Per-speed (per-bucket) live learner — the fix the bucket structure always implied.

The old FordBalLiveScale bucketed (desk, act) BY SPEED but then averaged all
buckets into ONE global multiplier. That can only correct a uniform offset; a
per-speed SHAPE error (over at low/high, under at mid — route 6a9fbd805a) averages
to ~1.0, so the global learner saw "no net error" and did nothing. AND its output
was ignored by bal_encode entirely.

This rewrite: each speed bucket learns its OWN correction, current_scale(v) returns
the per-speed value, and bal_encode applies it. So the mid-speed bucket can climb
to kill the wide while the low/high buckets (already at/over unity) stay put —
without adding the global loop gain that caused the weave.
"""
from opendbc.car.ford.lateral_bal import (
  FordBalLiveScale, bal_encode,
  FORD_BAL_LIVE_BOUND_LOW, FORD_BAL_LIVE_BOUND_HIGH, FORD_BAL_LIVE_TARGET,
)

MID_V = 12.5   # bucket [10,15) center — where the field "wide" lives
HI_V = 18.5    # bucket [15,22) center


def _drive(ls, v, delivery, n, desk=0.02):
  """Feed n stable-curve frames at speed v with realized act = delivery·desk."""
  for _ in range(n):
    ls.update(desk, delivery * desk, v, True, False)


def test_seed_targets_mid_speed_wide_only():
  """Fresh learner is seeded to give the under-delivering mid-speed band a head
  start, leaving low/high speed neutral (~1.0)."""
  ls = FordBalLiveScale()
  assert ls.current_scale(MID_V) > 1.0          # mid seeded up (fix wide)
  assert abs(ls.current_scale(5.0) - 1.0) < 1e-6  # low speed neutral


def test_learns_per_bucket_independently():
  """THE point: under-delivery at mid speed raises ONLY the mid bucket; the
  high-speed bucket is untouched. (The old global learner moved everything.)"""
  ls = FordBalLiveScale()
  hi_before = ls.current_scale(HI_V)
  mid_before = ls.current_scale(MID_V)
  _drive(ls, MID_V, 0.85, 4000)                 # persistent under-delivery at mid
  assert ls.current_scale(MID_V) > mid_before + 0.02, "mid bucket must rise"
  assert abs(ls.current_scale(HI_V) - hi_before) < 1e-6, "high bucket must NOT move"


def test_direction_under_raises_over_lowers():
  ls_u = FordBalLiveScale(); ls_o = FordBalLiveScale()
  base_u = ls_u.current_scale(HI_V); base_o = ls_o.current_scale(HI_V)
  _drive(ls_u, HI_V, 0.80, 4000)                # under-deliver -> command more
  _drive(ls_o, HI_V, 1.20, 4000)                # over-deliver -> command less
  assert ls_u.current_scale(HI_V) > base_u
  assert ls_o.current_scale(HI_V) < base_o


def test_converges_toward_target_delivery():
  """CLOSED loop: realized delivery = base·scale·desk (raising the scale raises
  delivery). The bucket scale converges so delivery -> TARGET, i.e. scale ->
  TARGET/base."""
  base = 0.85   # true plant delivery at scale=1 (under-delivering -> wide)
  desk = 0.02
  ls = FordBalLiveScale()
  for _ in range(20000):
    scale = ls.current_scale(HI_V)
    ls.update(desk, base * scale * desk, HI_V, True, False)  # act reflects applied scale
  expected = min(FORD_BAL_LIVE_TARGET / base, FORD_BAL_LIVE_BOUND_HIGH)
  assert abs(ls.current_scale(HI_V) - expected) < 0.03, f"{ls.current_scale(HI_V)} vs {expected}"


def test_scale_bounded():
  ls = FordBalLiveScale()
  _drive(ls, HI_V, 0.30, 30000)                 # wild under-delivery
  assert ls.current_scale(HI_V) <= FORD_BAL_LIVE_BOUND_HIGH + 1e-9
  ls2 = FordBalLiveScale()
  _drive(ls2, HI_V, 3.0, 30000)
  assert ls2.current_scale(HI_V) >= FORD_BAL_LIVE_BOUND_LOW - 1e-9


def test_gates_reject_straights_volatile_pressed():
  ls = FordBalLiveScale(); base = ls.current_scale(HI_V)
  for _ in range(4000): ls.update(0.0005, 0.0005, HI_V, True, False)  # straight
  assert abs(ls.current_scale(HI_V) - base) < 1e-9
  for _ in range(4000): ls.update(0.02, 0.017, HI_V, True, True)      # hands on
  assert abs(ls.current_scale(HI_V) - base) < 1e-9


def test_bal_encode_applies_per_speed_scale():
  _, c1_base, _ = bal_encode(0.02, MID_V, None, 1.0)
  _, c1_more, _ = bal_encode(0.02, MID_V, None, 1.10)
  _, c1_less, _ = bal_encode(0.02, MID_V, None, 0.90)
  assert c1_more > c1_base > c1_less
  assert bal_encode(0.02, MID_V, None, 1.0) == bal_encode(0.02, MID_V, None)
