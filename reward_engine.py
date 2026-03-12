"""
reward_engine.py
================
Absorbing-State Reward Engine

Reward function
---------------

Reward scaling — TICKS not NOTIONAL DOLLARS
    All terminal rewards are divided by (order_size × tick_size) so the
    final reward represents ticks of edge per share, not raw dollar PnL.

    Without this normalisation a 500-share order at a 0.01 tick has
    raw rewards ~100× larger than a 5-share order at the same edge.
    XGBoost's regression targets explode in scale across the dataset,
    the Bellman targets become unstable between iterations, and the
    Q-function diverges.  Tick-normalisation keeps rewards in [-5, +5]
    regardless of order size, making training stable.

Fill reward  (spread capture + 50ms mark-out, in ticks):
    Buy:   reward = (mid_50ms_after − limit_price) / (order_size × tick_size)
    Sell:  reward = (limit_price − mid_50ms_after) / (order_size × tick_size)

    If mid_50ms_after is NaN falls back to mid_at_fill (zero mark-out).

Cancel penalty  (100ms drift window, in ticks, 3× AS multiplier):
    Buy:   penalty = −max(0, mid_at_cancel − mid_100ms_ago) × 3 / tick_size
    Sell:  penalty = −max(0, mid_100ms_ago − mid_at_cancel) × 3 / tick_size

    The 3× adverse-selection multiplier makes the cancel penalty large
    enough that the agent learns cancelling is preferable to absorbing
    a toxic fill.  Dividing by tick_size (not order_size × tick_size)
    keeps cancel penalties comparable to fill rewards in tick units.

Intermediate states:
    reward = 0.0 strictly, unless hold penalty is enabled (see comment block).

Toggle: hold penalty
    Comment/uncomment one line — see _compute() below.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TRIG_FILL   = "FILL"
TRIG_CANCEL = "CANCEL"

MARKOUT_NS  = 50_000_000    # 50 ms forward window for fill mark-out
CANCEL_WINDOW_NS = 100_000_000  # 100 ms backward window for cancel penalty


class RewardEngine:
    """
    Parameters
    ----------
    tick_size : float or None
        Instrument tick size (used only as fallback for spread estimate).
    """

    def __init__(self, tick_size: Optional[float] = None, **kwargs) -> None:
        self._tick_size = tick_size

    def assign_rewards(self, transitions: List[dict]) -> List[dict]:
        """Populate 'reward' in-place on every transition. Returns same list."""
        for t in transitions:
            t["reward"] = self._compute(t)
        return transitions

    def summary(self, transitions: List[dict]) -> Dict:
        fills   = [t for t in transitions if t["terminal"] and t.get("trigger") == TRIG_FILL]
        cancels = [t for t in transitions if t["terminal"] and t.get("trigger") == TRIG_CANCEL]

        n_total  = len(transitions)
        n_fill   = len(fills)
        n_cancel = len(cancels)

        fill_rewards   = np.array([t["reward"] for t in fills],   dtype=float)
        cancel_rewards = np.array([t["reward"] for t in cancels], dtype=float)

        mean_fill   = float(np.mean(fill_rewards))   if n_fill   > 0 else np.nan
        mean_cancel = float(np.mean(cancel_rewards)) if n_cancel > 0 else np.nan

        total_fill_gain   = float(np.sum(fill_rewards[fill_rewards > 0]))  if n_fill   > 0 else 0.0
        total_cancel_loss = float(np.sum(np.abs(cancel_rewards)))          if n_cancel > 0 else 0.0
        n_cancel_zero     = int(np.sum(cancel_rewards == 0.0))             if n_cancel > 0 else 0

        cancel_drag_pct = (
            round(total_cancel_loss / total_fill_gain * 100, 2)
            if total_fill_gain > 0 else np.nan
        )

        return {
            "n_total":             n_total,
            "n_fill":              n_fill,
            "n_cancel":            n_cancel,
            "n_cancel_zero_drift": n_cancel_zero,
            "n_intermediate":      len(transitions) - n_fill - n_cancel,
            "fill_pct":            round(n_fill   / n_total * 100, 2) if n_total else np.nan,
            "cancel_pct":          round(n_cancel / n_total * 100, 2) if n_total else np.nan,
            "mean_fill_reward":    round(mean_fill,   6) if not np.isnan(mean_fill)   else np.nan,
            "mean_cancel_reward":  round(mean_cancel, 6) if not np.isnan(mean_cancel) else np.nan,
            "total_fill_gain":     round(total_fill_gain,   4),
            "total_cancel_loss":   round(total_cancel_loss, 4),
            # cancel_drag_pct: cancel losses as % of fill gains.
            # <50% healthy | ~100% break-even | >100% over-cancelling
            "cancel_drag_pct":     cancel_drag_pct if not np.isnan(cancel_drag_pct) else np.nan,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute(self, t: dict) -> float:

        # ── Non-terminal: optional hold penalty ──────────────────────────────
        # To enable: comment out the `return 0.0` line and uncomment the block.
        if not t["terminal"]:
            return 0.0   # ← comment this out to enable hold penalty below
            # --- HOLD PENALTY (uncomment block + comment line above) ---
            # side     = t.get("side", "")
            # mid_now  = t.get("mid",      np.nan)
            # mid_prev = t.get("mid_prev", np.nan)  # stamped by stage1
            # if np.isnan(mid_now) or np.isnan(mid_prev):
            #     return 0.0
            # return float(mid_now - mid_prev) if side == "B" else float(mid_prev - mid_now)
            # -----------------------------------------------------------

        trigger = t.get("trigger", "")
        side    = t.get("side", "")

        # ── Reward scaling: ticks not notional dollars ──────────────────────
        # Dividing by (order_size × tick_size) converts raw dollar PnL into
        # "ticks of edge" — a size-invariant unit.  Without this, a 500-share
        # order produces rewards 100× larger than a 5-share order at the same
        # edge, causing Bellman targets to explode in scale and the Q-function
        # to diverge after a few FQI iterations.
        order_size = float(t.get("order_size", 1))
        if order_size < 1:
            order_size = 1.0

        tick_size  = self._tick_size if self._tick_size else 0.01
        if tick_size <= 0:
            tick_size = 0.01

        fill_normaliser   = order_size * tick_size   # $/tick×share → ticks/share
        cancel_normaliser = tick_size                # cancel drift already per-share

        # ── FILL: spread capture + 50ms mark-out, in ticks ───────────────────
        if trigger == TRIG_FILL:
            limit_price    = t.get("limit_price",    np.nan)
            mid_at_fill    = t.get("mid",            np.nan)
            mid_50ms_after = t.get("mid_after_50ms", np.nan)

            if np.isnan(limit_price) or np.isnan(mid_at_fill):
                logger.warning("FILL missing limit_price/mid for vid=%s; reward=0.",
                               t.get("virtual_id", "?"))
                return 0.0

            if np.isnan(mid_50ms_after):
                mid_50ms_after = mid_at_fill

            if side == "B":
                raw = float(mid_50ms_after - limit_price)
            else:
                raw = float(limit_price - mid_50ms_after)

            # Divide by (order_size × tick_size): converts dollar gain to ticks.
            # A half-tick spread capture on any order size = reward of 0.5.
            return raw / fill_normaliser

        # ── CANCEL: 100ms drift penalty, in ticks, 3× AS multiplier ─────────
        # Dividing by tick_size keeps the unit in ticks.  The 3× multiplier
        # makes the toxic-flow signal dominant over a small drift: 2 ticks of
        # adverse momentum → penalty of 6.0 ticks, clearly worse than a clean
        # fill reward of ~0.5 ticks.
        ADVERSE_SELECTION_MULTIPLIER = 3.0

        if trigger == TRIG_CANCEL:
            mid_at_cancel  = t.get("mid",           np.nan)
            mid_100ms_ago  = t.get("mid_100ms_ago", np.nan)

            if np.isnan(mid_at_cancel) or np.isnan(mid_100ms_ago):
                return 0.0

            if side == "B":
                drift = max(0.0, mid_at_cancel - mid_100ms_ago)
            else:
                drift = max(0.0, mid_100ms_ago - mid_at_cancel)

            # Divide by tick_size: drift in price → penalty in ticks.
            return float(-drift) / cancel_normaliser * ADVERSE_SELECTION_MULTIPLIER

        logger.warning("Unknown terminal trigger '%s' for vid=%s; reward=0.",
                       trigger, t.get("virtual_id", "?"))
        return 0.0


def compute_and_summarise(
    transitions: List[dict],
    tick_size: Optional[float] = None,
    cancel_penalty_ticks: float = 0.0,   # kept for call-site backward-compat
) -> Tuple[List[dict], Dict]:
    engine      = RewardEngine(tick_size=tick_size)
    transitions = engine.assign_rewards(transitions)
    summary     = engine.summary(transitions)
    return transitions, summary

# backward-compat aliases
CANCEL_PRIORITY_PENALTY_TICKS = 0.0
CANCEL_PENALTY_TICKS          = 0.0