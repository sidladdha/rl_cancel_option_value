"""
shadow_tracker.py  v2
=====================
Counterfactual Shadow Order Wealth Framework

What this module answers
------------------------
For every strand-cancel fired by the RL tracker, we ask:
  "What would this order have earned if we had NOT cancelled?"

We birth a shadow order at the exact same price/side/size and let it live
in a second tracker that has no strand rule.  Three outcomes are possible:

  (a) Shadow fills within the SAME SESSION
          shadow_pnl = (mid_50ms_after - limit_price) * size   [buy]
                     = (limit_price - mid_50ms_after) * size   [sell]

  (b) Shadow NEVER fills (session ends first)
          shadow_pnl = (cancel_mid - birth_mid) * size  [buy]
                     = (birth_mid - cancel_mid) * size  [sell]
          clipped at 0 (no reward if mid moved in your favour).
          This is the real opportunity cost: the mid drifted against
          the order while it sat idle. If mid was flat, cost = 0.
          Falls back to -0.5 * tick * size if mid data is unavailable.

  (c) Shadow fill on a DIFFERENT SESSION  -> treated as (b), never filled.
          Cross-day fills are EXCLUDED entirely.  A buy at 150.00 that
          sits unfilled overnight and fills next morning is NOT a valid
          counterfactual - the book has reset, spreads have repriced, and
          the fill would be at stale queue priority.

Option Value formula
--------------------
  option_value = rl_pnl - shadow_pnl

  rl_pnl     : actual RL reward at cancel time (drift penalty, negative)
  shadow_pnl : fill reward if same-session fill,
               else -opportunity_cost * size

Interpretation
--------------
  option_value > 0  ->  RL cancel was CORRECT (cancelling beat holding)
  option_value = 0  ->  indifferent
  option_value < 0  ->  RL cancel was WRONG (holding would have been better)

Four-way cancel classification
-------------------------------
  CORRECT_NEVER_FILLED  option_value > 0, shadow never filled.
                         Dead-money avoidance.  Order would have sat idle
                         all day; cancel saved the opportunity cost.

  CORRECT_TOXIC_FILL    option_value > 0, shadow filled, AS score < 0.
                         Shadow filled but mid moved against us immediately
                         after fill - a toxic fill the RL correctly avoided.

  WRONG_CLEAN_FILL      option_value < 0, shadow filled, AS score >= 0.
                         Shadow filled cleanly; RL cancelled unnecessarily.
                         This is the "false positive cancel" to minimise.

  LUCKY_CANCEL          option_value >= 0, shadow filled, AS score >= 0.
                         Shadow filled cleanly BUT rl_pnl happened to be
                         less negative than shadow_pnl - coincidental win.

Output files
------------
  <prefix>_shadow_analysis.csv        one row per strand-cancelled order
  <prefix>_shadow_by_date_level.csv   breakdown by date x level_idx
  <prefix>_shadow_rl_vs_shadow.csv    daily RL PnL vs shadow PnL comparison
  <prefix>_adverse_selection.csv      AS analysis by date x level
  <prefix>_option_value_ccdf.png      log-CCDF + level bar chart
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_STRAND_DISABLED            = 10_000_000   # effectively inf - disables strand rule
OPPORTUNITY_COST_MULTIPLIER = 0.5          # * tick_size per share for unfilled shadow
MARKOUT_NS                  = 50_000_000   # 50 ms forward window


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ns_to_date(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def _classify(option_value: float, shadow_filled: bool, as_score: float) -> str:
    if not shadow_filled:
        return "CORRECT_NEVER_FILLED"
    as_neg = (not np.isnan(as_score)) and as_score < 0
    if option_value > 0 and as_neg:
        return "CORRECT_TOXIC_FILL"
    if option_value < 0:
        return "WRONG_CLEAN_FILL"
    return "LUCKY_CANCEL"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_shadow_analysis(
    transitions: List[dict],
    messages_df: pl.DataFrame,
    ts_ns_arr:   np.ndarray,
    mid_arr:     np.ndarray,
    tick_size:   float = 0.01,
) -> "pd.DataFrame":
    """
    Run shadow analysis for all strand-cancel events.

    Parameters
    ----------
    transitions : full transition list (rewards already assigned)
    messages_df : raw MBO polars DataFrame
    ts_ns_arr   : int64 timestamp array (from _build_mid_array)
    mid_arr     : float64 mid-price array aligned with ts_ns_arr
    tick_size   : instrument tick size (used for opportunity cost)

    Returns
    -------
    pd.DataFrame with one row per strand-cancelled order
    """
    import pandas as pd
    from virtual_order_tracker import VirtualOrderTracker, TRIG_FILL, TRIG_CANCEL
    import virtual_order_tracker as _vot_mod

    opp_cost_per_share = OPPORTUNITY_COST_MULTIPLIER * tick_size

    # ── 1. Collect strand-cancel terminals ───────────────────────────────────
    cancel_events: List[dict] = [
        t for t in transitions
        if t.get("terminal") and t.get("trigger") == TRIG_CANCEL
    ]
    if not cancel_events:
        logger.warning("shadow_tracker: no CANCEL transitions found; skipping.")
        return pd.DataFrame()

    logger.info("shadow_tracker: %d strand-cancel events to shadow.", len(cancel_events))

    # ── 2. Build session boundary map ────────────────────────────────────────
    SESSION_GAP_NS = 30 * 60 * 10 ** 9
    ts_local = messages_df["timestamp"].cast(pl.Int64).to_numpy()
    gaps     = np.diff(ts_local, prepend=ts_local[0])
    sess_ids = np.cumsum(gaps > SESSION_GAP_NS).astype(int)

    # session_id -> last timestamp in that session
    session_last_ns: Dict[int, int] = {}
    for sid, ts in zip(sess_ids, ts_local):
        session_last_ns[sid] = int(ts)

    logger.info("shadow_tracker: %d sessions detected.", len(session_last_ns))

    # ── 3. Index cancel events by timestamp ──────────────────────────────────
    cancel_by_ts: Dict[int, List[dict]] = {}
    for ce in cancel_events:
        cancel_by_ts.setdefault(int(ce["ts_ns"]), []).append(ce)

    shadow_to_cancel:  Dict[str, dict] = {}
    shadow_to_session: Dict[str, int]  = {}
    shadow_results:    Dict[str, dict] = {}

    # ── 4. Disable strand rule and replay stream ──────────────────────────────
    original_threshold = _vot_mod.STRAND_TICKS_THRESHOLD
    _vot_mod.STRAND_TICKS_THRESHOLD = _STRAND_DISABLED

    shadow_tracker = VirtualOrderTracker(queue_change_threshold=0.10)
    rows_list      = list(messages_df.iter_rows(named=True))
    prev_session   = int(sess_ids[0])

    for i, row in enumerate(rows_list):
        ts_ns       = int(ts_local[i])
        cur_session = int(sess_ids[i])

        # ── Session boundary: expire all shadow orders from prev session ──────
        # This is the cross-day contamination fix.
        # Any shadow order still alive when its session ends is "never filled".
        if cur_session != prev_session:
            for svid in list(shadow_to_cancel.keys()):
                if shadow_to_session.get(svid) != prev_session:
                    continue
                res = shadow_results.get(svid)
                if res and not res["shadow_filled"]:
                    sz        = res["order_size"]
                    # Opportunity cost = actual mid drift from birth to cancel.
                    # This is the real cost of holding a stranded order: the
                    # mid moved against you by this much while the order sat idle.
                    birth_mid  = float(res.get("birth_mid", np.nan))
                    cancel_mid = float(res.get("cancel_mid", np.nan))
                    side       = res["side"]
                    if not np.isnan(birth_mid) and not np.isnan(cancel_mid):
                        if side == "B":
                            # mid fell → order value eroded
                            opp = min(0.0, cancel_mid - birth_mid) * sz
                        else:
                            # mid rose → short order value eroded
                            opp = min(0.0, birth_mid - cancel_mid) * sz
                    else:
                        opp = -(opp_cost_per_share * sz)   # fallback
                    res["shadow_pnl"]   = round(opp, 6)
                    res["option_value"] = round(res["rl_pnl"] - opp, 6)
                    res["expired_eod"]  = True
                # Only call cancel_order if still live in tracker
                if svid in shadow_tracker._virtual_orders:
                    shadow_tracker.cancel_order(svid, ts_ns)
                shadow_to_cancel.pop(svid, None)
                shadow_to_session.pop(svid, None)
            shadow_tracker.reset_book()
            prev_session = cur_session

        # ── Place shadow orders for RL cancels at this timestamp ──────────────
        if ts_ns in cancel_by_ts:
            for ce in cancel_by_ts[ts_ns]:
                ce_session = int(ce.get("session_id", cur_session))
                if ce_session != cur_session:
                    continue
                svid = shadow_tracker.place_order(
                    side  = ce["side"],
                    price = float(ce["limit_price"]),
                    size  = int(ce.get("order_size", 1)),
                    ts_ns = ts_ns,
                )
                sz  = int(ce.get("order_size", 1))
                opp = -(opp_cost_per_share * sz)
                shadow_to_cancel[svid]  = ce
                shadow_to_session[svid] = ce_session
                shadow_results[svid] = {
                    "vid_rl":            ce["virtual_id"],
                    "cancel_ts_ns":      ts_ns,
                    "date":              _ns_to_date(ts_ns),
                    "session_id":        ce_session,
                    "side":              ce["side"],
                    "price":             float(ce["limit_price"]),
                    "level_idx":         int(ce.get("level_idx", 0)) + 1,
                    "order_size":        sz,
                    "rl_pnl":            float(ce.get("reward", 0.0)),
                    # mid at birth and at cancel — used to compute real opp cost
                    "birth_mid":         float(ce.get("birth_mid", np.nan)),
                    "cancel_mid":        float(ce.get("mid",       np.nan)),
                    "shadow_filled":     False,
                    "shadow_fill_ts_ns": np.nan,
                    "shadow_pnl":        round(opp, 6),
                    "option_value":      round(float(ce.get("reward", 0.0)) - opp, 6),
                    "as_score":          np.nan,
                    "expired_eod":       False,
                    "cancel_type":       "",
                }

        # ── Process message through shadow tracker ────────────────────────────
        for st in shadow_tracker.on_message(row):
            svid = st["virtual_id"]
            if not st["terminal"] or st.get("trigger") != TRIG_FILL:
                continue
            if svid not in shadow_results:
                continue

            # Cross-session fill guard
            fill_ts  = int(st["ts_ns"])
            fill_idx = max(0, int(np.searchsorted(ts_local, fill_ts, side="left")) - 1)
            fill_session = int(sess_ids[fill_idx])
            if fill_session != shadow_to_session.get(svid, cur_session):
                logger.debug(
                    "shadow_tracker: cross-session fill rejected svid=%s "
                    "(cancel_sess=%d fill_sess=%d)",
                    svid[:8], shadow_to_session.get(svid, -1), fill_session,
                )
                continue

            ce  = shadow_to_cancel[svid]
            res = shadow_results[svid]
            lp  = float(ce["limit_price"])
            sz  = int(ce.get("order_size", 1))
            mid_at_fill = float(st.get("mid", np.nan))

            idx50    = int(np.searchsorted(ts_ns_arr, fill_ts + MARKOUT_NS, side="left"))
            mid_50ms = (float(mid_arr[idx50])
                        if idx50 < len(mid_arr) and not np.isnan(mid_arr[idx50])
                        else mid_at_fill)

            side = ce["side"]
            shadow_pnl = ((mid_50ms - lp) if side == "B" else (lp - mid_50ms)) * sz
            as_score   = ((mid_50ms - mid_at_fill) if side == "B"
                          else (mid_at_fill - mid_50ms)) if not np.isnan(mid_at_fill) else np.nan

            res["shadow_filled"]     = True
            res["shadow_fill_ts_ns"] = float(fill_ts)
            res["shadow_pnl"]        = round(shadow_pnl, 6)
            res["option_value"]      = round(res["rl_pnl"] - shadow_pnl, 6)
            res["as_score"]          = round(as_score, 6) if not np.isnan(as_score) else np.nan
            res["expired_eod"]       = False

    # ── Expire remaining open orders at end of stream ────────────────────────
    for svid, res in shadow_results.items():
        if not res["shadow_filled"]:
            sz         = res["order_size"]
            birth_mid  = float(res.get("birth_mid", np.nan))
            cancel_mid = float(res.get("cancel_mid", np.nan))
            side       = res["side"]
            if not np.isnan(birth_mid) and not np.isnan(cancel_mid):
                opp = (min(0.0, cancel_mid - birth_mid) * sz if side == "B"
                       else min(0.0, birth_mid - cancel_mid) * sz)
            else:
                opp = -(opp_cost_per_share * sz)
            res["shadow_pnl"]   = round(opp, 6)
            res["option_value"] = round(res["rl_pnl"] - opp, 6)
            res["expired_eod"]  = True

    _vot_mod.STRAND_TICKS_THRESHOLD = original_threshold

    # ── 5. Build DataFrame and classify ──────────────────────────────────────
    import pandas as pd

    df = pd.DataFrame(list(shadow_results.values()))
    if df.empty:
        return df

    df["cancel_type"] = df.apply(
        lambda r: _classify(
            r["option_value"], r["shadow_filled"],
            r["as_score"] if not pd.isna(r["as_score"]) else np.nan,
        ),
        axis=1,
    )

    # ── 6. Overall summary log ────────────────────────────────────────────────
    n        = len(df)
    n_filled = int(df["shadow_filled"].sum())
    n_never  = n - n_filled
    mean_ov  = df["option_value"].mean()
    pct_corr = float((df["option_value"] > 0).mean() * 100)
    ct       = df["cancel_type"].value_counts().to_dict()
    filled_  = df[df["shadow_filled"]]
    mean_as  = filled_["as_score"].mean()          if n_filled > 0 else np.nan
    pct_tox  = float((filled_["as_score"] < 0).mean() * 100) if n_filled > 0 else 0.0

    logger.info(
        "\n"
        "=== SHADOW ORDER / COUNTERFACTUAL WEALTH ANALYSIS ===\n"
        "Opportunity cost (unfilled shadow) = %.5f per share (%.1fx tick_size)\n\n"
        "Shadow outcomes (cross-day fills EXCLUDED):\n"
        "  Total cancels tracked  : %d\n"
        "  Shadow filled same-day : %d (%.1f%%)\n"
        "  Shadow never filled    : %d (%.1f%%)\n\n"
        "Option Value = rl_pnl - shadow_pnl\n"
        "  Mean option value      : %+.6f\n"
        "  Cancel CORRECT         : %.1f%%\n"
        "  Cancel WRONG           : %.1f%%\n\n"
        "Cancel type breakdown:\n"
        "  CORRECT_NEVER_FILLED   : %d\n"
        "  CORRECT_TOXIC_FILL     : %d\n"
        "  WRONG_CLEAN_FILL       : %d\n"
        "  LUCKY_CANCEL           : %d\n\n"
        "Adverse Selection (filled shadows only):\n"
        "  Mean AS score          : %+.6f  (negative = toxic fill)\n"
        "  Pct toxic fills        : %.1f%%",
        opp_cost_per_share, OPPORTUNITY_COST_MULTIPLIER,
        n, n_filled, n_filled / n * 100, n_never, n_never / n * 100,
        mean_ov, pct_corr, 100 - pct_corr,
        ct.get("CORRECT_NEVER_FILLED", 0), ct.get("CORRECT_TOXIC_FILL", 0),
        ct.get("WRONG_CLEAN_FILL",     0), ct.get("LUCKY_CANCEL",       0),
        mean_as if not np.isnan(mean_as) else 0.0, pct_tox,
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Breakdown: date x level_idx
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_breakdown(
    shadow_df: "pd.DataFrame",
    outdir: "Path",
    save_prefix: str,
) -> "pd.DataFrame":
    """
    Full breakdown split by date x level_idx.

    Columns per cell:
      n_cancels, n_shadow_filled, n_never_filled, fill_rate_pct
      total_rl_pnl, total_shadow_pnl, total_option_value
      mean_option_value, pct_correct
      n_correct_never, n_correct_toxic, n_wrong_clean, n_lucky
      mean_as_score, pct_toxic_fills
      verdict

    Also writes daily RL-vs-shadow PnL comparison table.
    """
    import pandas as pd
    from pathlib import Path

    if shadow_df.empty:
        logger.warning("generate_full_breakdown: empty DataFrame.")
        return pd.DataFrame()

    def _cell(grp):
        ov  = grp["option_value"].to_numpy()
        rl  = grp["rl_pnl"].to_numpy()
        sh  = grp["shadow_pnl"].to_numpy()
        ct  = grp["cancel_type"].value_counts().to_dict()
        n   = len(grp)
        nf  = int(grp["shadow_filled"].sum())
        as_ = grp.loc[grp["shadow_filled"], "as_score"].dropna().to_numpy()
        pct = float((ov > 0).mean() * 100)

        if n < 3:
            verdict = "too few - not meaningful"
        elif pct >= 70:
            verdict = "STRONG CANCEL EDGE"
        elif pct >= 50:
            verdict = "MILD CANCEL EDGE"
        elif ct.get("WRONG_CLEAN_FILL", 0) / max(n, 1) > 0.4:
            verdict = "OVER-CANCELLING - missing clean fills"
        else:
            verdict = "MARGINAL"

        return pd.Series({
            "n_cancels":          n,
            "n_shadow_filled":    nf,
            "n_never_filled":     n - nf,
            "fill_rate_pct":      round(nf / n * 100, 1),
            "total_rl_pnl":       round(float(rl.sum()), 5),
            "total_shadow_pnl":   round(float(sh.sum()), 5),
            "total_option_value": round(float(ov.sum()), 5),
            "mean_option_value":  round(float(ov.mean()), 6),
            "pct_correct":        round(pct, 1),
            "n_correct_never":    ct.get("CORRECT_NEVER_FILLED", 0),
            "n_correct_toxic":    ct.get("CORRECT_TOXIC_FILL",   0),
            "n_wrong_clean":      ct.get("WRONG_CLEAN_FILL",     0),
            "n_lucky":            ct.get("LUCKY_CANCEL",         0),
            "mean_as_score":      round(float(as_.mean()), 6) if len(as_) else np.nan,
            "pct_toxic_fills":    round(float((as_ < 0).mean() * 100), 1) if len(as_) else np.nan,
            "verdict":            verdict,
        })

    breakdown = (
        shadow_df
        .groupby(["date", "level_idx"], sort=True)
        .apply(_cell)
        .reset_index()
        .sort_values(["date", "level_idx"])
    )

    path = Path(outdir) / f"{save_prefix}_shadow_by_date_level.csv"
    breakdown.to_csv(str(path), index=False)
    logger.info(
        "\n=== Shadow Breakdown: Date x Level ===\n%s\nSaved -> %s",
        breakdown.to_string(index=False), path,
    )

    # ── Daily RL PnL vs shadow PnL ────────────────────────────────────────────
    daily = (
        shadow_df
        .groupby("date", sort=True)
        .agg(
            n_cancels         =("rl_pnl",        "count"),
            total_rl_pnl      =("rl_pnl",        "sum"),
            total_shadow_pnl  =("shadow_pnl",    "sum"),
            total_option_value=("option_value",  "sum"),
            pct_correct       =("option_value",
                                lambda x: round(float((x > 0).mean() * 100), 1)),
            n_shadow_filled   =("shadow_filled", "sum"),
        )
        .reset_index()
    )
    daily["rl_beat_shadow"]          = daily["total_option_value"] > 0
    daily["cumulative_rl_pnl"]       = daily["total_rl_pnl"].cumsum()
    daily["cumulative_shadow_pnl"]   = daily["total_shadow_pnl"].cumsum()
    daily["cumulative_option_value"] = daily["total_option_value"].cumsum()

    daily_path = Path(outdir) / f"{save_prefix}_shadow_rl_vs_shadow.csv"
    daily.to_csv(str(daily_path), index=False)
    logger.info(
        "\n=== Daily: RL PnL vs Shadow PnL ===\n"
        "total_rl_pnl    : actual P&L from RL cancel decisions (all negative)\n"
        "total_shadow_pnl: what we would have made by holding (fill or opp-cost)\n"
        "total_option_value = rl - shadow  (positive = cancelling was better)\n\n%s\n"
        "Saved -> %s",
        daily.to_string(index=False), daily_path,
    )

    return breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Log-CCDF + level bar chart
# ─────────────────────────────────────────────────────────────────────────────

def generate_ccdf_plot(
    shadow_df: "pd.DataFrame",
    outdir: "Path",
    save_prefix: str,
) -> None:
    """
    Two-panel figure:
      Left  : Log-CCDF of option value (Figure 4 style from Kwan-Philip).
      Right : Median option value per price level bar chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping CCDF plot.")
        return

    import pandas as pd
    from pathlib import Path

    if shadow_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: log-CCDF
    ax = axes[0]
    for label, mask, colour in [
        ("All cancels",                 pd.Series([True] * len(shadow_df)), "#1f77b4"),
        ("Shadow filled (non-trivial)", shadow_df["shadow_filled"],          "#d62728"),
    ]:
        ov = shadow_df.loc[mask, "option_value"].dropna().sort_values().to_numpy()
        if len(ov) < 2:
            continue
        ccdf = 1.0 - np.arange(1, len(ov) + 1) / len(ov)
        ax.semilogy(ov, ccdf, label=label, color=colour, linewidth=1.8)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.9, alpha=0.6,
               label="Break-even")
    ax.set_xlabel("Option Value  (rl_pnl - shadow_pnl)", fontsize=10)
    ax.set_ylabel("P(X > v)  [log scale]", fontsize=10)
    ax.set_title("Log-CCDF: Cancel Option Value", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Right: median option value per level
    ax2 = axes[1]
    lv = (shadow_df.groupby("level_idx")["option_value"]
          .median().reset_index().sort_values("level_idx"))
    colours = ["#2ca02c" if m >= 0 else "#d62728" for m in lv["option_value"]]
    ax2.bar(lv["level_idx"].astype(str), lv["option_value"],
            color=colours, alpha=0.8)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Price Level (1=BBO)", fontsize=10)
    ax2.set_ylabel("Median Option Value", fontsize=10)
    ax2.set_title("Median Option Value by Level\n"
                  "(green=cancel correct, red=should have held)", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = Path(outdir) / f"{save_prefix}_option_value_ccdf.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("CCDF + level chart saved -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Adverse Selection summary: date x level
# ─────────────────────────────────────────────────────────────────────────────

def adverse_selection_summary(
    shadow_df: "pd.DataFrame",
    outdir: "Path",
    save_prefix: str,
) -> "pd.DataFrame":
    """
    Adverse Selection summary split by date x level.

    Only rows where shadow_filled=True have a real AS score.

    AS score > 0 : mid moved in your favour after fill  (clean fill)
    AS score < 0 : mid moved against you after fill     (TOXIC fill -
                   the RL was RIGHT to cancel)

    Key columns:
      pct_toxic_fills    : if >50%, cancelling at this level/date was justified
      pct_cancel_correct : % of ALL cancels (incl never-filled) with OV > 0
      mean_excess_return : mean option value across all cancels
    """
    import pandas as pd
    from pathlib import Path

    filled = shadow_df[shadow_df["shadow_filled"]].copy()
    if filled.empty:
        logger.warning("adverse_selection_summary: no filled shadows.")
        return pd.DataFrame()

    def _agg(grp):
        as_arr = grp["as_score"].dropna().to_numpy()
        ov_arr = grp["option_value"].dropna().to_numpy()
        # For pct_cancel_correct use all cancels (not just filled)
        all_at_cell = shadow_df[
            (shadow_df["date"]      == grp["date"].iloc[0]) &
            (shadow_df["level_idx"] == grp["level_idx"].iloc[0])
        ]
        all_ov = all_at_cell["option_value"].to_numpy()
        pct_tox = float((as_arr < 0).mean() * 100) if len(as_arr) else np.nan
        return pd.Series({
            "n_filled_shadows":    len(grp),
            "n_total_cancels":     len(all_at_cell),
            "fill_rate_pct":       round(len(grp) / max(len(all_at_cell), 1) * 100, 1),
            "mean_as_score":       round(float(as_arr.mean()),  6) if len(as_arr) else np.nan,
            "pct_toxic_fills":     round(pct_tox, 1),
            "mean_option_value":   round(float(ov_arr.mean()), 6) if len(ov_arr) else np.nan,
            "pct_cancel_correct":  round(float((all_ov > 0).mean() * 100), 1)
                                   if len(all_ov) else np.nan,
            "interpretation": (
                "CANCEL JUSTIFIED - >50% held fills toxic"
                if (len(as_arr) > 0 and (as_arr < 0).mean() > 0.5)
                else "OVER-CANCELLING - most held fills were clean"
            ),
        })

    summary = (
        filled
        .groupby(["date", "level_idx"], sort=True)
        .apply(_agg)
        .reset_index()
        .sort_values(["date", "level_idx"])
    )

    logger.info(
        "\n=== Adverse Selection: Date x Level (filled shadows only) ===\n"
        "AS < 0 = toxic fill (market moved against us after fill)\n"
        "High pct_toxic_fills -> RL was correct to cancel at this level/date\n\n%s",
        summary.to_string(index=False),
    )

    path = Path(outdir) / f"{save_prefix}_adverse_selection.csv"
    summary.to_csv(str(path), index=False)
    logger.info("Adverse selection saved -> %s", path)
    return summary