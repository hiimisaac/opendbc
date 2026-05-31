"""
Ford CAN-FD lateral control — PSCM-inverse "bal" allocation.

Inputs:
  desired_curvature (1/m) from the planner — already contains turning intent
  plus drift correction.

Output: a path polynomial (c0, c1, c2, curvature_rate) for PSCM's
LateralMotionControl2 message such that the curvature PSCM delivers at the
wheels matches the requested desired_curvature as closely as possible.

Method:
  1. The PSCM polynomial-to-curvature transfer was characterized empirically
     across ~49,000 active driving frames; per-speed-bin coefficients
     α(v), β(v), γ(v) such that act_κ ≈ α·c0 + β·c1 + γ·c2.
  2. To deliver |desk|, pick a utilization u = |desk|/(β·L_c1 + γ·L_c2) and
     emit c1 = sign(desk)·u·L_c1, c2 = sign(desk)·u·L_c2. Channels with
     non-positive coefficient at that speed are skipped (they would actively
     fight desk — observed e.g. γ<0 below 3 m/s on the platforms measured).
  3. If c1+c2 alone are insufficient (u_c12 > 1), the c0 channel is brought
     in as overflow at the same utilization to extend authority. This keeps
     c0 silent in normal driving (where α is small and uncertain) and only
     uses it at the hardware ceiling.
  4. A per-platform scalar multiplies β and γ to absorb vehicle-to-vehicle
     calibration differences (mass/wheelbase/PSCM-cal trim). Default is 1.0
     for any platform without an explicit entry.
  5. An optional online estimator (FordBalLiveScale) observes (desk, act_κ)
     on stable curves and slowly adjusts a per-vehicle live multiplier
     bounded ±15% of the platform default. Persists via openpilot Params
     when available; runs in-memory otherwise.

Curvature rate (4th polynomial slot, LatCtlCrv_NoRate2_Actl) is the rate
of change of curvature over distance, dκ/dx = (Δdesk/Δt) / v_ego, clipped
to the DBC range ±0.001023 1/m². Computed by the carcontroller.
"""
from __future__ import annotations
from typing import Optional, Tuple


# ─── PSCM transfer function (regression table) ───────────────────────────────
# Empirical multi-variate regression over 49,117 active frames / 31 routes /
# 93 segments. Intercept dropped (it captured road-crown bias, ≤7e-3).
#
#  (v_max m/s,   α,         β,         γ,         R²)
FORD_PSCM_REGRESSION: Tuple[Tuple[float, float, float, float, float], ...] = (
  ( 3.0,  0.00903,  0.01708, -0.15194, 0.45),  # parking — degenerate, low R²
  ( 6.0,  0.00914,  0.06149,  0.39633, 0.78),
  (10.0,  0.00134,  0.08842,  0.61726, 0.83),
  (15.0,  0.00157,  0.06843,  0.64632, 0.87),
  (22.0,  0.00070,  0.05117,  0.79698, 0.93),  # highway — best fit
  (35.0, -0.00191,  0.04402,  0.63223, 0.89),
)

# Per-platform multiplier on β/γ to absorb per-vehicle calibration differences.
# scale > 1.0 → shrinks bal's u allocation → less c1+c2 sent (cancels over-delivery).
# scale < 1.0 → grows u → more c1+c2 sent (cancels under-delivery).
# Empty by default — all platforms start at 1.0 and the live estimator
# discovers each vehicle's actual offset from there.
FORD_PSCM_PLATFORM_SCALE: dict[str, float] = {}
FORD_PSCM_PLATFORM_SCALE_DEFAULT = 1.0

# Per-channel DBC clip magnitudes (LateralMotionControl2 signal ranges).
FORD_BAL_LC0 = 5.11    # m
FORD_BAL_LC1 = 0.5235  # rad
FORD_BAL_LC2 = 0.02    # 1/m

# ─── WBal allocation (ported from sp-dev-c3, identified on this Lightning) ────
# c1-DOMINANT (well-damped path-angle) + a SMALL low-speed c0 (offset) centering
# trim, sized by MEASURED per-channel gains (act per unit oriented wire) so
# realized ≈ desk. c2 dropped (sticky accumulator). Gains from a 12-agent gain-
# identification over a 626K-frame pool + current-firmware run, cross-validated
# and reproducing the field over/under outcomes. Two field fixes baked ON:
#   - g_c1 overshoot trim (cancels the slight low-speed sustained over-delivery)
#   - c0 straight deadband (silences the offset channel on near-straight desk,
#     where its tiny g_c0 divisor amplifies noise into a slow ping-pong weave)
FORD_WBAL_V_BP       = (4.0, 8.0, 12.0)
FORD_WBAL_C0_FRAC    = (0.30, 0.18, 0.05)   # c0 delivery share; ~off by 12 m/s
FORD_WBAL_GC0_V      = (3.0, 5.0, 8.0, 12.0, 18.0)
FORD_WBAL_GC0        = (0.0040, 0.0035, 0.0010, 0.0004, 0.00015)
FORD_WBAL_GC1_V      = (3.0, 5.0, 8.0, 12.0, 18.0)
# Workflow-identified values. NOTE: a ×0.92 boost (smaller g_c1 = more c1 per
# desk = more loop gain) was tried to cancel ~8% under-delivery, but the 12-cycle
# analysis of route 6a9fbd805a showed the sustained-curve weave is a high-gain
# closed-loop limit cycle (act LEADS desk by 0.54s; wheel slams ±15deg on a tiny
# desk ripple) and the boost made it WORSE. Reverted to the identified values to
# drop the loop gain back. The residual ~8% wide is the lesser evil vs the weave.
FORD_WBAL_GC1        = (0.125, 0.139, 0.102, 0.078, 0.057)
FORD_WBAL_GC1_TRIM_V = (3.0, 5.0, 8.0)      # baked ON: raise g_c1 to kill overshoot
FORD_WBAL_GC1_TRIM   = (1.08, 1.03, 1.00)
FORD_WBAL_C0_DB_LO   = 0.004                # baked ON: c0 deadband fade range (1/m)
FORD_WBAL_C0_DB_HI   = 0.008
FORD_WBAL_C0_RATE    = 6.0                  # m/s — c0 slew (applied in carcontroller)

# ─── Live estimator constants ─────────────────────────────────────────────────
# Slow first-order filter on the live scalar, bounded ±25% from platform default.
# Robustness layers modeled on selfdrive/locationd/torqued.py — including a
# ramping filter decay that converges fast initially and slows as data builds,
# so single bad routes can't yank a well-converged value.
FORD_BAL_LIVE_VERSION         = "v2"   # v2: per-speed-bucket scale (was single scalar)
FORD_BAL_LIVE_BOUND_LOW       = 0.75   # ±25% from platform default
FORD_BAL_LIVE_BOUND_HIGH      = 1.25
# Target DELIVERY (act/desk) each bucket converges to. Deliberately just UNDER
# unity: full delivery (1.0) sits the loop gain on the marginal-stability edge
# where the sustained-curve weave lives, so we hold a hair below it.
FORD_BAL_LIVE_TARGET          = 0.97
# Per-bucket seed (len == #buckets). The field per-speed delivery on route
# 6a9fbd805a was over at low/high speed, UNDER (~0.86) only in the mid band —
# so seed only the mid bucket up (immediate wide relief) and let the learner
# refine the rest from neutral. Buckets: [3-6),[6-10),[10-15),[15-22),[22-35).
FORD_BAL_LIVE_SEED            = (1.0, 1.0, 1.05, 1.0, 1.0)
FORD_BAL_LIVE_MIN_DECAY       = 50.0   # initial filter decay (alpha ≈ 0.02 — fast)
FORD_BAL_LIVE_MAX_DECAY       = 500.0  # final filter decay   (alpha ≈ 0.002 — stable)
FORD_BAL_LIVE_DECAY_STEP      = 1.0    # decay increment per successful update
FORD_BAL_LIVE_MIN_BUCKET_FRAMES = 100  # fresh samples in a bucket before it trims (clean estimate)
FORD_BAL_LIVE_BUCKETS_V       = (3.0, 6.0, 10.0, 15.0, 22.0, 35.0)
FORD_BAL_LIVE_DESK_MIN        = 0.003  # |desk| floor for inclusion
FORD_BAL_LIVE_V_MIN           = 3.0    # m/s — exclude parking
FORD_BAL_LIVE_DESK_VAR_MAX    = 0.002  # std over 1 s window; reject volatile desk
FORD_BAL_LIVE_DESK_BUF        = 20     # ~1 s @ 20 Hz fresh data
FORD_BAL_LIVE_PERSIST_FRAMES  = 6000   # write to Params every ~60 s @ 100 Hz

# Soft import — opendbc as a library doesn't require openpilot. If Params
# isn't available the estimator runs in-memory and resets each drive.
try:
  from openpilot.common.params import Params  # type: ignore
  _HAS_PARAMS = True
except Exception:
  Params = None  # type: ignore
  _HAS_PARAMS = False


def _clip(x: float, lo: float, hi: float) -> float:
  return lo if x < lo else (hi if x > hi else x)


def _interp(x: float, xs, ys) -> float:
  """Plain-python 1-D linear interp (no numpy — keep lateral_bal a pure lib)."""
  if x <= xs[0]:
    return ys[0]
  if x >= xs[-1]:
    return ys[-1]
  for i in range(1, len(xs)):
    if x < xs[i]:
      t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
  return ys[-1]


def pscm_response_at(v_ego: float, fingerprint: Optional[str],
                     live_scale: float = 1.0) -> Tuple[float, float, float]:
  """Return (α, β, γ) for the speed bin containing v_ego, with β and γ
  scaled by (platform_scale × live_scale). α is not scaled because it is
  the worst-identified regression coefficient (R²=-0.17 on c0-only data)
  and was driving slight over-delivery in field tests when scaled."""
  base = FORD_PSCM_PLATFORM_SCALE.get(fingerprint, FORD_PSCM_PLATFORM_SCALE_DEFAULT)
  scale = base * live_scale
  a = b = g = 0.0
  for v_max, av, bv, gv, _r2 in FORD_PSCM_REGRESSION:
    if v_ego < v_max:
      a, b, g = av, bv, gv
      break
  else:
    _, a, b, g, _ = FORD_PSCM_REGRESSION[-1]
  return a, b * scale, g * scale


def bal_encode(desired_curvature: float, v_ego: float,
               fingerprint: Optional[str], live_scale: float = 1.0
               ) -> Tuple[float, float, float]:
  """WBal: c1-DOMINANT + small low-speed c0 (offset) centering trim, sized by
  MEASURED per-channel gains so realized ≈ desk (g_c0·c0 + g_c1·c1 = desk). c2=0.
  Both field fixes baked ON (g_c1 overshoot trim, c0 straight deadband). Spills a
  saturated channel's unmet delivery onto the other.

  live_scale (RECONNECTED, per-speed): the bucket's delivery trim from
  FordBalLiveScale.current_scale(v). >1 → the learner saw under-delivery at this
  speed, so command proportionally MORE (and vice-versa). This self-corrects the
  per-speed wide. fingerprint unused. Returns INTERNAL sign convention (positive
  desk → positive c0/c1); the carcontroller negates before packing onto CAN."""
  if abs(desired_curvature) < 1e-9:
    return 0.0, 0.0, 0.0
  s = 1.0 if desired_curvature > 0.0 else -1.0
  mag = abs(desired_curvature)

  w0 = _interp(v_ego, FORD_WBAL_V_BP, FORD_WBAL_C0_FRAC)
  # c0 straight deadband (baked ON): fade c0's delivery share to 0 as desk ->
  # straight; c1 takes the slack so total delivery stays = desk (no step).
  gate = (mag - FORD_WBAL_C0_DB_LO) / (FORD_WBAL_C0_DB_HI - FORD_WBAL_C0_DB_LO)
  w0 *= _clip(gate, 0.0, 1.0)

  g0 = max(_interp(v_ego, FORD_WBAL_GC0_V, FORD_WBAL_GC0), 1e-5)
  g1 = max(_interp(v_ego, FORD_WBAL_GC1_V, FORD_WBAL_GC1), 1e-5)
  g1 *= _interp(v_ego, FORD_WBAL_GC1_TRIM_V, FORD_WBAL_GC1_TRIM)  # overshoot trim ON

  ls = _clip(live_scale, FORD_BAL_LIVE_BOUND_LOW, FORD_BAL_LIVE_BOUND_HIGH)
  c0 = w0 * mag / g0 * ls       # g_c0·c0 = w0·mag·live_scale
  c1 = (1.0 - w0) * mag / g1 * ls  # g_c1·c1 = (1-w0)·mag·live_scale
  c0c = c0 if c0 < FORD_BAL_LC0 else FORD_BAL_LC0
  c1c = c1 if c1 < FORD_BAL_LC1 else FORD_BAL_LC1
  unmet = (c0 - c0c) * g0 + (c1 - c1c) * g1
  if unmet > 1e-9:
    if c1c < FORD_BAL_LC1:
      c1c = min(FORD_BAL_LC1, c1c + unmet / g1)
    elif c0c < FORD_BAL_LC0:
      c0c = min(FORD_BAL_LC0, c0c + unmet / g0)
  return s * c0c, s * c1c, 0.0


class FordBalLiveScale:
  """Per-SPEED online estimator. Each speed bucket observes (desk, act_κ) on
  stable curves and slowly trims ITS OWN delivery multiplier toward
  FORD_BAL_LIVE_TARGET / measured_gain, bounded ±25%. current_scale(v) returns
  the per-speed trim, interpolated smoothly across bucket centers.

  Why per-bucket: the field error is a per-SPEED shape (over at low/high, under
  at mid). A single global multiplier averages those to ~1.0 and does nothing;
  only an independent per-bucket trim can raise the mid band (kill the wide)
  while leaving the low/high bands (already at/over unity) alone — so it does
  NOT add the global loop gain that drives the weave.

  Persistence via Params if available; in-memory otherwise. Payload:
      "FINGERPRINT:VERSION:live0,live1,...:decay0,decay1,..."
  Discarded on fingerprint or version change (v2 = per-bucket format)."""
  def __init__(self):
    self._params = Params() if _HAS_PARAMS else None
    self._fingerprint: Optional[str] = None
    self._n = len(FORD_BAL_LIVE_BUCKETS_V) - 1
    self._centers = [(FORD_BAL_LIVE_BUCKETS_V[i] + FORD_BAL_LIVE_BUCKETS_V[i + 1]) / 2.0
                     for i in range(self._n)]
    self._live = list(FORD_BAL_LIVE_SEED)
    # Per-bucket ramping filter decay: converges fast early, slows as data builds.
    self._decay = [FORD_BAL_LIVE_MIN_DECAY] * self._n
    self._bucket_act_sum = [0.0] * self._n
    self._bucket_desk_sum = [0.0] * self._n
    self._bucket_count = [0] * self._n
    self._desk_buf: list[float] = []
    self._frame = 0
    self._persist_frame = 0

  def _bucket(self, v: float) -> Optional[int]:
    for i in range(self._n):
      if FORD_BAL_LIVE_BUCKETS_V[i] <= v < FORD_BAL_LIVE_BUCKETS_V[i + 1]:
        return i
    return None

  # --- persistence ---------------------------------------------------------
  def _restore(self, fingerprint: str) -> None:
    self._fingerprint = fingerprint
    self._live = list(FORD_BAL_LIVE_SEED)
    self._decay = [FORD_BAL_LIVE_MIN_DECAY] * self._n
    if self._params is None:
      return
    try:
      raw = self._params.get("FordBalLiveScale")
      if not raw:
        return
      text = raw.decode() if isinstance(raw, bytes) else raw
      parts = text.split(":")
      if len(parts) < 3 or parts[0] != fingerprint or parts[1] != FORD_BAL_LIVE_VERSION:
        return
      lv = [float(x) for x in parts[2].split(",")]
      if len(lv) == self._n:
        self._live = [_clip(x, FORD_BAL_LIVE_BOUND_LOW, FORD_BAL_LIVE_BOUND_HIGH) for x in lv]
      if len(parts) >= 4:
        dv = [float(x) for x in parts[3].split(",")]
        if len(dv) == self._n:
          self._decay = [_clip(x, FORD_BAL_LIVE_MIN_DECAY, FORD_BAL_LIVE_MAX_DECAY) for x in dv]
    except Exception:
      self._live = list(FORD_BAL_LIVE_SEED)
      self._decay = [FORD_BAL_LIVE_MIN_DECAY] * self._n

  def _persist(self) -> None:
    if self._params is None or self._fingerprint is None:
      return
    try:
      lv = ",".join(f"{x:.4f}" for x in self._live)
      dv = ",".join(f"{x:.1f}" for x in self._decay)
      self._params.put_nonblocking("FordBalLiveScale",
                                   f"{self._fingerprint}:{FORD_BAL_LIVE_VERSION}:{lv}:{dv}")
    except Exception:
      # UnknownKeyName if host openpilot hasn't registered the key — that's
      # OK, the estimator continues running in-memory.
      pass

  # --- public ---------------------------------------------------------------
  def current_scale(self, v: float, fingerprint: Optional[str] = None) -> float:
    if fingerprint is not None and fingerprint != self._fingerprint:
      self._restore(fingerprint)
    return _interp(float(v), self._centers, self._live)

  def update(self, desk: float, act_k: float, v: float,
             lat_active: bool, steering_pressed: bool) -> None:
    """Tick once per carcontroller frame. Cheap when the gates reject
    (most non-curve frames)."""
    self._frame += 1
    self._desk_buf.append(desk)
    if len(self._desk_buf) > FORD_BAL_LIVE_DESK_BUF:
      self._desk_buf.pop(0)

    # Persistence throttle always fires while running so a value gets written
    # even on routes with few curves.
    if self._frame - self._persist_frame >= FORD_BAL_LIVE_PERSIST_FRAMES:
      self._persist_frame = self._frame
      self._persist()

    # Frame gates
    if not lat_active or steering_pressed:
      return
    if v < FORD_BAL_LIVE_V_MIN:
      return
    if abs(desk) < FORD_BAL_LIVE_DESK_MIN:
      return
    if len(self._desk_buf) < FORD_BAL_LIVE_DESK_BUF:
      return
    # Stdev gate — reject volatile desk (planner hunting / lane change). Note a
    # mild weave passes (std ≪ MAX); its symmetric over/under averages out in
    # the bucket sums, so the learner reads the true mean delivery.
    mean = sum(self._desk_buf) / len(self._desk_buf)
    var = sum((x - mean) ** 2 for x in self._desk_buf) / len(self._desk_buf)
    if var > FORD_BAL_LIVE_DESK_VAR_MAX * FORD_BAL_LIVE_DESK_VAR_MAX:
      return

    b = self._bucket(v)
    if b is None:
      return
    self._bucket_act_sum[b] += act_k * (1.0 if desk > 0 else -1.0)
    self._bucket_desk_sum[b] += abs(desk)
    self._bucket_count[b] += 1

    # Per-bucket trim: when THIS bucket has enough fresh data, nudge ITS scale
    # toward the value that would make its delivery == TARGET. Independent of
    # every other bucket — that's the whole point.
    if self._bucket_count[b] >= FORD_BAL_LIVE_MIN_BUCKET_FRAMES and self._bucket_desk_sum[b] > 1e-6:
      gain = self._bucket_act_sum[b] / self._bucket_desk_sum[b]  # observed delivery (incl. current scale)
      if 0.3 < gain < 3.0:  # sanity
        target = _clip(FORD_BAL_LIVE_TARGET * self._live[b] / gain,
                       FORD_BAL_LIVE_BOUND_LOW, FORD_BAL_LIVE_BOUND_HIGH)
        alpha = 1.0 / self._decay[b]
        self._live[b] = _clip((1.0 - alpha) * self._live[b] + alpha * target,
                              FORD_BAL_LIVE_BOUND_LOW, FORD_BAL_LIVE_BOUND_HIGH)
        self._decay[b] = min(self._decay[b] + FORD_BAL_LIVE_DECAY_STEP, FORD_BAL_LIVE_MAX_DECAY)
      # Decay this bucket's accumulators so stale data fades.
      self._bucket_act_sum[b] *= 0.75
      self._bucket_desk_sum[b] *= 0.75
      self._bucket_count[b] = int(self._bucket_count[b] * 0.75)
