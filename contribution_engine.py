"""
contribution_engine.py
======================
Primary Analysis Driver — Counterfactual Contribution Analysis

This module is the heart of the evaluation pipeline.  It takes the flat
event lists from VirtualOrderTracker (Standard + Shadow modes) and
computes two normalised contribution metrics per filled order.

All outputs are expressed in TICKS (price movement / tick_size) so results are
comparable across different order sizes and instruments.

NOTE: fill contributions (toxic_ticks, reposition_ticks, total_ticks) retain the
original normaliser of order_size × tick_size for backward compatibility.
Cancel-value outputs use normaliser = tick_size only, making v_cancel_ticks
size-independent (see calculate_cancel_value for full derivation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TOXIC FLOW CONTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The 100 ms mark-out for every fill.

    Raw (dollars):
        BUY:   raw_toxic = fill_price − mid_at_T+100ms
        SELL:  raw_toxic = mid_at_T+100ms − fill_price

    Normalised (ticks):
        toxic_ticks = raw_toxic / (order_size × tick_size)

    Interpretation:
        < 0  You bought and the mid fell (or sold and mid rose) — adverse
             selection.  The fill was toxic; you paid more than fair value
             within 100 ms.
        > 0  Mid moved in your favour — spread capture / informed flow.
        = 0  Neutral fill; no post-fill price movement.

    A strategy with mean_toxic_ticks < −0.5 is being systematically
    picked off by informed order flow.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. REPOSITIONING VALUE (Reposition Contribution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compares the repositioned order's outcome vs the shadow order at the
original birth price.

    Requires pairing:
        Standard event  → fill at repositioned limit_price
        Shadow event    → fill (or no fill) at shadow_price

    Four cases:

    Case A  (standard filled, shadow did NOT fill):
        Reposition saved the trade.  The original price was too far from
        the market to attract a fill; repositioning was beneficial.
        reposition_ticks = +toxic_ticks  (full fill value credited)

    Case B  (both filled):
        Priority Loss.  Both prices filled, but repositioning moved the
        order to a worse queue position.  The shadow would have filled at
        the original price (better for a buy, worse for a sell).
        BUY:   reposition_ticks = (fill_price − shadow_fill_price) / (size × tick)
        SELL:  reposition_ticks = (shadow_fill_price − fill_price) / (size × tick)
        Typically negative — repositioning cost queue priority.

    Case C  (shadow filled, standard did NOT):
        Repositioning overshot the market.  The original price would have
        filled naturally; the repositioned price never traded.
        reposition_ticks = −|shadow_toxic_ticks|  (opportunity missed)

    Case D  (neither filled — both cancelled):
        No comparison possible.
        reposition_ticks = 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. ADVERSE SELECTION SCORE (AS Score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Derived from shadow fills only (the counterfactual).

    as_score = shadow_fill_toxic_ticks
             = (shadow_fill_price − shadow_mid_post_100ms) / (size × tick)  [buy]

    A consistently negative AS score means the shadow (passive, no
    repositioning) was also adversely selected — the problem is structural,
    not just a repositioning artefact.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output columns per fill
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    virtual_id, date, session_id, side, level_idx
    fill_price, shadow_fill_price, repositioned
    mid_post_100ms, shadow_mid_post_100ms
    order_size, tick_size, normaliser         (= order_size × tick_size)
    toxic_ticks                               (Toxic Flow Contribution)
    reposition_ticks                          (Reposition Contribution)
    total_ticks                               (toxic + reposition)
    as_score_ticks                            (Adverse Selection Score)
    reposition_case                           (A/B/C/D)
    state                                     (5-feature vector from standard event)

Cancel-value output (calculate_cancel_value):
    virtual_id, date, session_id, side, level_idx
    cancel_mid, birth_mid, mid_100ms_ago
    rl_pnl_ticks     — cancel_mid vs shadow_fill_price in ticks (shadow-fill case)
                        or birth→cancel mid drift in ticks (no-fill fallback)
    shadow_filled, shadow_fill_price, shadow_mid_post_100ms
    shadow_pnl_ticks — mark-out of shadow fill 100ms later (in ticks)
    v_cancel_ticks   — rl_pnl − shadow_pnl; simplifies to
                        (cancel_mid − shadow_mid_post) / tick_size [buy]
    v_cancel_bps     — v_cancel as basis points of order notional; SIZE-INDEPENDENT
                        = v_cancel_ticks × tick_size / cancel_mid × 10_000
    v_cancel_dollars — total $ pnl of this cancel = v_cancel_ticks × tick × order_size
    order_size, normaliser (= tick_size only)
    queue_pressure_ratio, qp_frac_gate
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TRIG_FILL   = "FILL"
TRIG_CANCEL = "CANCEL"


def _ns_to_date(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


class ContributionEngine:
    """
    Computes tick-normalised Toxic Flow, Reposition, and AS contributions.

    Usage
    -----
    engine = ContributionEngine(tick_size=0.01)

    # Pass both standard and shadow terminal events together.
    # The engine pairs them by virtual_id automatically.
    rows = engine.calculate(standard_events, shadow_events)

    import pandas as pd
    df = pd.DataFrame(rows)
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive; got {tick_size}")
        self._tick_size = tick_size

    # ── Primary entry point ───────────────────────────────────────────────────

    def calculate(
        self,
        standard_events: List[dict],
        shadow_events:   Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Calculate contributions for all filled standard orders.

        Parameters
        ----------
        standard_events : terminal events from VirtualOrderTracker(mode="standard")
        shadow_events   : terminal events from VirtualOrderTracker(mode="shadow")
                          If None, reposition_ticks and as_score are NaN.

        Returns
        -------
        List of contribution dicts, one per FILL in standard_events.
        """
        # Index standard fills by virtual_id
        std_fills: Dict[str, dict] = {
            t["virtual_id"]: t
            for t in standard_events
            if t.get("terminal") and t.get("trigger") == TRIG_FILL
        }

        # Index shadow fills by virtual_id
        shd_fills: Dict[str, dict] = {}
        if shadow_events:
            shd_fills = {
                t["virtual_id"]: t
                for t in shadow_events
                if t.get("terminal") and t.get("trigger") == TRIG_FILL
            }

        rows = []
        for vid, std_t in std_fills.items():
            shd_t = shd_fills.get(vid)   # may be None if shadow didn't fill
            row   = self._compute(std_t, shd_t)
            if row:
                # Back-populate reward on the standard transition for legacy compat
                std_t["reward"] = row["total_ticks"]
                rows.append(row)

        if rows:
            tox  = [r["toxic_ticks"]       for r in rows if not np.isnan(r["toxic_ticks"])]
            rep  = [r["reposition_ticks"]   for r in rows if not np.isnan(r["reposition_ticks"])]
            tot  = [r["total_ticks"]        for r in rows if not np.isnan(r["total_ticks"])]
            asc  = [r["as_score_ticks"]     for r in rows if not np.isnan(r["as_score_ticks"])]
            logger.info(
                "ContributionEngine: %d fills | "
                "mean_toxic=%.4f ticks | mean_reposition=%.4f ticks | "
                "mean_total=%.4f ticks | mean_as_score=%.4f ticks",
                len(rows),
                np.mean(tox) if tox else np.nan,
                np.mean(rep) if rep else np.nan,
                np.mean(tot) if tot else np.nan,
                np.mean(asc) if asc else np.nan,
            )
        else:
            logger.warning("ContributionEngine: no fills found in standard_events.")

        return rows

    # ── Internal computation ──────────────────────────────────────────────────

    def _compute(self, std: dict, shd: Optional[dict]) -> dict:
        """Compute all contributions for one standard fill."""
        side         = std.get("side", "")
        fill_price   = float(std.get("exec_price",    np.nan))
        mid_post     = float(std.get("mid_post_100ms", np.nan))
        shadow_price      = float(std.get("shadow_price",      fill_price))
        repositioned      = bool(std.get("repositioned",       False))
        order_size        = max(int(std.get("order_size",      1)), 1)
        ts_ns             = int(std.get("ts_ns", std.get("birth_ts_ns", 0)))
        birth_ts_ns       = int(std.get("birth_ts_ns",         ts_ns))
        reposition_ts_ns  = int(std.get("reposition_ts_ns",    0))
        level_idx         = int(std.get("level_idx", 0)) + 1   # 1-based for readability

        if np.isnan(fill_price):
            return {}

        tick     = self._tick_size
        norm     = order_size * tick   # normaliser: dollars → ticks per share

        # ── 1. Toxic Flow Contribution ─────────────────────────────────────
        if not np.isnan(mid_post):
            raw_toxic = (fill_price - mid_post) if side == "B" else (mid_post - fill_price)
            toxic_ticks = raw_toxic / norm
        else:
            toxic_ticks = np.nan

        # ── 2. Reposition Contribution ─────────────────────────────────────
        shd_fill_price = float(shd["exec_price"])    if shd else np.nan
        shd_mid_post   = float(shd["mid_post_100ms"]) if shd else np.nan

        if not repositioned:
            # Not repositioned — shadow_price == fill_price, no comparison needed
            reposition_ticks = 0.0
            reposition_case  = "D_NO_REPOSITION"

        elif shd and not np.isnan(shd_fill_price):
            # Case B: both standard and shadow filled
            # Priority loss: repositioned order filled at a worse queue position
            if side == "B":
                raw_rep = fill_price - shd_fill_price
            else:
                raw_rep = shd_fill_price - fill_price
            reposition_ticks = raw_rep / norm
            reposition_case  = "B_BOTH_FILLED_PRIORITY_LOSS"

        elif not np.isnan(fill_price) and (shd is None or np.isnan(shd_fill_price)):
            # Case A: standard filled, shadow did NOT
            # Repositioning saved the trade — full fill value credited
            reposition_ticks = toxic_ticks if not np.isnan(toxic_ticks) else 0.0
            reposition_case  = "A_REPOSITION_SAVED_FILL"

        else:
            # Case C: shadow filled but standard did not (should not happen
            # inside this branch since we only enter for std_fills, but
            # guard defensively)
            if not np.isnan(shd_mid_post):
                raw_shd_tox = ((shd_fill_price - shd_mid_post) if side == "B"
                               else (shd_mid_post - shd_fill_price))
                reposition_ticks = -abs(raw_shd_tox / norm)
            else:
                reposition_ticks = 0.0
            reposition_case = "C_SHADOW_FILLED_STANDARD_DID_NOT"

        # ── 3. Adverse Selection Score ─────────────────────────────────────
        # Measured on shadow fill only — the counterfactual passive order
        if shd and not np.isnan(shd_fill_price) and not np.isnan(shd_mid_post):
            raw_as = ((shd_fill_price - shd_mid_post) if side == "B"
                      else (shd_mid_post - shd_fill_price))
            as_score_ticks = raw_as / norm
        else:
            as_score_ticks = np.nan

        total_ticks = (
            (toxic_ticks if not np.isnan(toxic_ticks) else 0.0)
            + reposition_ticks
        )

        # ── Case-split columns ─────────────────────────────────────────────
        # These allow summarise_pipeline to compute clean per-case means
        # without mixing repositioned (BBO) fills with passive (birth-price) fills.
        #
        #   toxic_ticks_passive : toxic_ticks for Case D orders (no reposition).
        #                         Reflects true adverse selection at original queue depth.
        #   toxic_ticks_repo    : toxic_ticks for Case A orders (repositioned to BBO).
        #                         Lower than passive because fill_price ≈ mid at repo time.
        #   total_ticks_passive : total_ticks for Case D (= toxic_ticks, no reposition credit).
        #   total_ticks_repo    : total_ticks for Case A (= 2 × toxic_ticks_repo by design).
        #   reposition_ticks_caseA : reposition_ticks for Case A only (NaN for Case D).
        #                         Gives the true mean reposition value without D-order dilution.
        is_case_A = (reposition_case == "A_REPOSITION_SAVED_FILL")
        is_case_D = (reposition_case == "D_NO_REPOSITION")
        toxic_ticks_passive  = toxic_ticks  if is_case_D else np.nan
        toxic_ticks_repo     = toxic_ticks  if is_case_A else np.nan
        total_ticks_passive  = total_ticks  if is_case_D else np.nan
        total_ticks_repo     = total_ticks  if is_case_A else np.nan
        repo_ticks_caseA     = reposition_ticks if is_case_A else np.nan

        # repo_to_fill_ms: elapsed time between the reposition event and the fill.
        # Only meaningful for Case A (repositioned + filled).
        # reposition_ts_ns = 0 means the order was never repositioned.
        # ts_ns here = fill_ts_ns (the terminal FILL event timestamp).
        if is_case_A and reposition_ts_ns > 0 and ts_ns > reposition_ts_ns:
            repo_to_fill_ms = (ts_ns - reposition_ts_ns) / 1e6   # nanoseconds → ms
        else:
            repo_to_fill_ms = np.nan

        # birth_to_fill_ms: total order lifetime for all fills.
        birth_to_fill_ms = (ts_ns - birth_ts_ns) / 1e6 if ts_ns > birth_ts_ns else np.nan

        return {
            "virtual_id":           std.get("virtual_id", ""),
            "date":                 _ns_to_date(ts_ns),
            "session_id":           int(std.get("session_id", 0)),
            "side":                 side,
            "level_idx":            level_idx,
            # Prices
            "fill_price":           round(fill_price,    5),
            "shadow_price":         round(shadow_price,  5),
            "shadow_fill_price":    round(shd_fill_price, 5) if not np.isnan(shd_fill_price) else np.nan,
            "repositioned":         repositioned,
            "mid_post_100ms":       round(mid_post,      5) if not np.isnan(mid_post)      else np.nan,
            "shadow_mid_post_100ms":round(shd_mid_post,  5) if not np.isnan(shd_mid_post)  else np.nan,
            # Sizing
            "order_size":           order_size,
            "tick_size":            tick,
            "normaliser":           round(norm, 6),
            # Contributions (all in ticks) — blended across cases
            "toxic_ticks":          round(toxic_ticks,        6) if not np.isnan(toxic_ticks)        else np.nan,
            "reposition_ticks":     round(reposition_ticks,   6),
            "total_ticks":          round(total_ticks,         6),
            "as_score_ticks":       round(as_score_ticks,      6) if not np.isnan(as_score_ticks)     else np.nan,
            "reposition_case":      reposition_case,
            # Case-split columns — NaN when not applicable to that case.
            # Use these instead of blended means in level_summary to avoid
            # repo contamination.  summarise_pipeline uses nanmean over each.
            "toxic_ticks_passive":  round(toxic_ticks_passive,  6) if not np.isnan(toxic_ticks_passive)  else np.nan,
            "toxic_ticks_repo":     round(toxic_ticks_repo,     6) if not np.isnan(toxic_ticks_repo)     else np.nan,
            "total_ticks_passive":  round(total_ticks_passive,  6) if not np.isnan(total_ticks_passive)  else np.nan,
            "total_ticks_repo":     round(total_ticks_repo,     6) if not np.isnan(total_ticks_repo)     else np.nan,
            "reposition_ticks_caseA": round(repo_ticks_caseA,  6) if not np.isnan(repo_ticks_caseA)     else np.nan,
            # Timing metrics
            # repo_to_fill_ms : ms between reposition and fill (Case A only; NaN otherwise).
            #   Measures how quickly the market came to the repositioned BBO price.
            #   summarise_pipeline reports mean/median/p95 of this per level.
            # birth_to_fill_ms: total order lifetime from placement to fill (all cases).
            "repo_to_fill_ms":        round(repo_to_fill_ms,   3) if not np.isnan(repo_to_fill_ms)       else np.nan,
            "birth_to_fill_ms":       round(birth_to_fill_ms,  3) if not np.isnan(birth_to_fill_ms)      else np.nan,
            # State vector for downstream analysis
            "state":                std.get("state"),
        }


    def calculate_cancel_value(
        self,
        standard_events: List[dict],
        shadow_events:   Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Calculate V_cancel = RL_PnL - Shadow_PnL for every strand-cancelled
        standard order.

        Parameters
        ----------
        standard_events : all terminal events from the standard tracker
        shadow_events   : all terminal events from the shadow tracker
                          (matched by virtual_id)

        Returns
        -------
        List of cancel-value row dicts, one per strand-cancelled order.
        """
        from datetime import datetime, timezone

        # Index shadow terminals by virtual_id — FILL takes priority over CANCEL.
        #
        # Bug context: when a standard order is QPR-cancelled, _retire_std()
        # immediately force-cancels the paired shadow.  But on that same message,
        # the shadow tracker may have already detected a fill (price_cross) and
        # appended a TRIG_FILL event to shd_events.  _retire_std then appends a
        # second terminal TRIG_CANCEL for the same vid, immediately after.
        #
        # A plain dict comprehension iterates in append order, so the CANCEL
        # (appended last) silently overwrites the FILL — making shadow_filled=False
        # for every cancel row, even when the shadow genuinely filled.
        #
        # Fix: iterate once, keeping the FILL if we see one; only store CANCEL
        # if no terminal event has been seen yet for that vid.
        shd_by_vid: Dict[str, dict] = {}
        if shadow_events:
            for t in shadow_events:
                if not t.get("terminal"):
                    continue
                vid = t["virtual_id"]
                # FILL takes priority: if this vid already has a terminal event,
                # only overwrite it if the new event is a FILL (fill beats cancel).
                if vid not in shd_by_vid or t.get("trigger") == TRIG_FILL:
                    shd_by_vid[vid] = t

        rows = []
        for t in standard_events:
            if not t.get("terminal"):
                continue
            if t.get("trigger") != TRIG_CANCEL:
                continue

            vid        = t["virtual_id"]
            side       = t.get("side", "")
            order_size = max(int(t.get("order_size", 1)), 1)
            tick       = self._tick_size
            # FIX-21a: normalise by tick_size only (not order_size × tick_size).
            # Using order_size × tick made v_cancel a per-share metric and caused
            # 96× inflation for size-1 orders (norm=0.01 instead of ~2.0).
            # With norm=tick, v_cancel_ticks = price_move_in_ticks, size-independent
            # and directly comparable across all orders.  Dollar value is still
            # recoverable as: v_cancel_dollars = v_cancel_ticks * tick * order_size.
            norm       = tick
            ts_ns      = int(t.get("ts_ns", t.get("birth_ts_ns", 0)))
            level_idx  = int(t.get("level_idx", 0)) + 1   # 1-based

            birth_mid    = float(t.get("birth_mid",    np.nan))
            cancel_mid   = float(t.get("mid",          np.nan))   # mid at cancel moment
            mid_100ms    = float(t.get("mid_100ms_ago", np.nan))  # mid 100ms before cancel
            qp_ratio     = float(t.get("queue_pressure_ratio", np.nan))
            qp_frac_gate = bool(t.get("qp_q_frac_gate", False))

            # ── Shadow outcome ─────────────────────────────────────────────
            shd = shd_by_vid.get(vid)
            shadow_filled      = False
            shadow_fill_price  = np.nan
            shadow_mid_post    = np.nan
            shadow_pnl_ticks   = 0.0   # default: shadow never filled → no opp cost

            if shd and shd.get("trigger") == TRIG_FILL:
                shadow_filled     = True
                shadow_fill_price = float(shd.get("exec_price",     np.nan))
                shadow_mid_post   = float(shd.get("mid_post_100ms", np.nan))

                if not np.isnan(shadow_fill_price) and not np.isnan(shadow_mid_post):
                    # shadow_pnl: mark-out of the shadow fill 100ms later.
                    # Positive = price moved in our favour after shadow filled (good fill).
                    # Negative = adverse selection — price moved against us (bad fill).
                    raw_shd = ((shadow_mid_post - shadow_fill_price) if side == "B"
                               else (shadow_fill_price - shadow_mid_post))
                    shadow_pnl_ticks = raw_shd / norm
                # else: shadow filled but no post-fill mid → shadow_pnl stays 0

            # ── RL_PnL ────────────────────────────────────────────────────
            # FIX-21b: rl_pnl is the opportunity value of the cancel relative to
            # what the shadow would have earned.
            #
            # CASE 1 — shadow filled:
            #   rl_pnl = (cancel_mid − shadow_fill_price) / tick  [buy]
            #          = (shadow_fill_price − cancel_mid) / tick  [sell]
            #   This is: "how far above (buy) or below (sell) the shadow's fill price
            #   was the mid at cancel time?"  Positive = you got out at a better level
            #   than the shadow eventually filled at.
            #   v_cancel then simplifies to:
            #       (cancel_mid − shadow_mid_post) / tick  [buy]
            #   = total mid decline from your cancel to 100ms after shadow filled.
            #   This is the cleanest measure of cancel quality.
            #
            # CASE 2 — shadow never filled:
            #   No fill comparison available.  Fall back to birth→cancel mid drift
            #   (mid_100ms_ago if available, else cancel_mid), normalised by tick.
            #   Positive = mid moved in our favour while the order was live, so the
            #   cancel preserved unrealised gain.
            if shadow_filled and not np.isnan(shadow_fill_price) and not np.isnan(cancel_mid):
                raw_rl = ((cancel_mid - shadow_fill_price) if side == "B"
                          else (shadow_fill_price - cancel_mid))
                rl_pnl_ticks = raw_rl / norm
            else:
                # No shadow fill — use mid drift over order lifetime as proxy.
                # Prefer mid_100ms_ago (tighter window: captures the market state
                # just before the QPR cancel fired) over cancel_mid.
                ref_mid = mid_100ms if not np.isnan(mid_100ms) else cancel_mid
                if not np.isnan(birth_mid) and not np.isnan(ref_mid):
                    raw_rl = (ref_mid - birth_mid) if side == "B" else (birth_mid - ref_mid)
                    rl_pnl_ticks = raw_rl / norm
                else:
                    rl_pnl_ticks = 0.0   # no reference data available

            # ── V_cancel ──────────────────────────────────────────────────
            # v_cancel = rl_pnl − shadow_pnl
            # When shadow filled with full mid data this simplifies to:
            #   (cancel_mid − shadow_mid_post) / tick  [buy side]
            # = total mid move from cancel time to 100ms after shadow settled.
            v_cancel_ticks = rl_pnl_ticks - shadow_pnl_ticks

            # ── bps and dollar outputs ─────────────────────────────────────
            # v_cancel_bps: cancel value as basis points of order notional.
            # Formula: bps = v_cancel_ticks * tick_size / cancel_mid * 10_000
            # Derivation: v_cancel_dollars = v_cancel_ticks * tick  (total $ move in 1 tick)
            #             But that's per-tick, not per-order.  Per-order dollar value:
            #               v_cancel_dollars = v_cancel_ticks * tick * order_size
            #             Notional = order_size * cancel_mid
            #             bps = v_cancel_dollars / notional * 10_000
            #                 = (v_cancel_ticks * tick * order_size) / (order_size * cancel_mid) * 10_000
            #                 = v_cancel_ticks * tick / cancel_mid * 10_000   ← order_size cancels
            # v_cancel_bps is therefore SIZE-INDEPENDENT and directly comparable
            # across all orders.  1 bps = 0.01% of notional.
            if not np.isnan(cancel_mid) and cancel_mid > 0:
                v_cancel_bps     = v_cancel_ticks * tick / cancel_mid * 10_000
                v_cancel_dollars = v_cancel_ticks * tick * order_size   # total $ pnl this order
            else:
                v_cancel_bps     = np.nan
                v_cancel_dollars = np.nan

            date_str = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")

            rows.append({
                "virtual_id":            vid,
                "date":                  date_str,
                "session_id":            int(t.get("session_id", 0)),
                "side":                  side,
                "level_idx":             level_idx,
                "cancel_mid":            round(cancel_mid,   5) if not np.isnan(cancel_mid)  else np.nan,
                "birth_mid":             round(birth_mid,    5) if not np.isnan(birth_mid)   else np.nan,
                "mid_100ms_ago":         round(mid_100ms,    5) if not np.isnan(mid_100ms)   else np.nan,
                "rl_pnl_ticks":          round(rl_pnl_ticks,          6),
                "shadow_filled":         shadow_filled,
                "shadow_fill_price":     round(shadow_fill_price, 5) if not np.isnan(shadow_fill_price) else np.nan,
                "shadow_mid_post_100ms": round(shadow_mid_post,   5) if not np.isnan(shadow_mid_post)   else np.nan,
                "shadow_pnl_ticks":      round(shadow_pnl_ticks,       6),
                "v_cancel_ticks":        round(v_cancel_ticks,          6),
                # v_cancel_bps: cancel value as basis points of order notional.
                # Size-independent — order_size cancels in the derivation.
                # Directly answers: "what fraction of this order's value did the cancel protect?"
                "v_cancel_bps":          round(v_cancel_bps,            4) if not np.isnan(v_cancel_bps)     else np.nan,
                # v_cancel_dollars: total $ PnL of this cancel decision (size-dependent).
                # = v_cancel_ticks * tick_size * order_size
                "v_cancel_dollars":      round(v_cancel_dollars,        6) if not np.isnan(v_cancel_dollars) else np.nan,
                "order_size":            order_size,
                # normaliser is now tick_size only (not order_size × tick_size).
                # Dollar value = v_cancel_ticks * normaliser * order_size.
                "normaliser":            round(tick, 6),
                # QPR diagnostic fields
                "queue_pressure_ratio":  round(qp_ratio, 6) if qp_ratio == qp_ratio else np.nan,
                "qp_frac_gate":          qp_frac_gate,
            })

        n_shd_fills = sum(1 for r in rows if r["shadow_filled"])
        mean_vc_ticks = float(np.mean([r["v_cancel_ticks"] for r in rows])) if rows else np.nan
        mean_vc_bps   = float(np.nanmean([r["v_cancel_bps"]   for r in rows])) if rows else np.nan
        logger.info(
            "calculate_cancel_value: %d cancels | shadow filled=%d (%.1f%%) | "
            "mean_v_cancel=%+.4f ticks (%+.4f bps)",
            len(rows), n_shd_fills,
            n_shd_fills / max(len(rows), 1) * 100,
            mean_vc_ticks, mean_vc_bps,
        )
        return rows

    # ── Convenience summary ───────────────────────────────────────────────────

    def event_summary(self, standard_events: List[dict]) -> dict:
        """Quick fill/cancel count summary over standard events."""
        fills   = [t for t in standard_events if t.get("terminal") and t.get("trigger") == TRIG_FILL]
        cancels = [t for t in standard_events if t.get("terminal") and t.get("trigger") == TRIG_CANCEL]
        n       = len(standard_events)
        return {
            "n_total":    n,
            "n_fill":     len(fills),
            "n_cancel":   len(cancels),
            "fill_pct":   round(len(fills)   / max(n, 1) * 100, 2),
            "cancel_pct": round(len(cancels) / max(n, 1) * 100, 2),
        }


# ── Backward-compat shims ─────────────────────────────────────────────────────

class RewardEngine(ContributionEngine):
    """Legacy shim — ContributionEngine is the replacement."""
    def assign_rewards(self, transitions: List[dict]) -> List[dict]:
        rows = self.calculate(transitions)
        for t in transitions:
            if np.isnan(float(t.get("reward", np.nan))):
                t["reward"] = 0.0
        return transitions


def compute_and_summarise(transitions, tick_size=None, cancel_penalty_ticks=0.0):
    engine = ContributionEngine(tick_size=tick_size or 0.01)
    rows   = engine.calculate(transitions)
    return transitions, engine.event_summary(transitions)


CANCEL_PRIORITY_PENALTY_TICKS = 0.0
CANCEL_PENALTY_TICKS          = 0.0