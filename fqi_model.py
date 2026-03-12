"""
fqi_model.py
============
Counterfactual Contribution Reporting — replaces FQI training

The Fitted Q-Iteration reinforcement learning logic has been completely
removed.  This file is now a pure reporting module that aggregates the
tick-normalised contribution results from ContributionEngine into a
structured summary table grouped by LOB level.

No XGBoost.  No Bellman equation.  No walk-forward training loops.
Runs in under one second on any dataset.

Primary output table (one row per level × side):
    mean_toxic_ticks       Toxic Flow Contribution
    mean_reposition_ticks  Repositioning Value
    mean_as_score_ticks    Adverse Selection Score
    mean_total_ticks       Combined total
    verdict                Plain-English interpretation

Backward-compatible constants (STATE_COLS, ACTION_HOLD, etc.) and shim
classes (WalkForwardValidator, build_transition_df) are preserved so any
existing call-sites continue to work without modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Backward-compat constants ─────────────────────────────────────────────────
STATE_COLS     = ["q_frac", "ln_shares_ahead", "OFI", "spread", "ln_order_size"]
ACTION_HOLD    = 0
ACTION_CANCEL  = 1
ACTIONS        = [ACTION_HOLD, ACTION_CANCEL]
GAMMA_DEFAULT  = 0.99
N_ITER_DEFAULT = 8
WINSOR_LOW     = 1
WINSOR_HIGH    = 99


# ─────────────────────────────────────────────────────────────────────────────
# Primary reporting function
# ─────────────────────────────────────────────────────────────────────────────

def build_level_summary(contribution_rows: List[dict]) -> pd.DataFrame:
    """
    Build the primary reporting table: one row per (level_idx, side).

    Columns
    -------
    level_idx             : 1-based LOB level (1 = BBO)
    side                  : B (buy) | A (ask/sell)
    n_fills               : number of filled orders at this level
    n_repositioned        : fills where 10-tick reposition fired
    pct_repositioned      : n_repositioned / n_fills * 100

    mean_toxic_ticks      : Mean Toxic Flow Contribution in ticks
                            negative = adverse selection
    std_toxic_ticks       : standard deviation
    pct_toxic_negative    : % of fills with toxic_ticks < 0

    mean_reposition_ticks : Mean Repositioning Value in ticks
                            negative = priority loss from repositioning
    pct_reposition_loss   : % of repositioned fills with reposition < 0

    mean_as_score_ticks   : Mean Adverse Selection Score in ticks
                            (measured on shadow fills)
                            negative = shadow passively adversely selected
    pct_as_negative       : % of shadow fills with as_score < 0

    mean_total_ticks      : toxic + reposition combined

    case_A_pct            : % Case A (reposition saved trade)
    case_B_pct            : % Case B (both filled, priority loss)
    case_C_pct            : % Case C (shadow filled, standard did not)
    case_D_pct            : % Case D (no repositioning)

    verdict               : plain-English interpretation
    """
    if not contribution_rows:
        logger.warning("build_level_summary: no contribution rows.")
        return pd.DataFrame()

    df = pd.DataFrame(contribution_rows)

    def _agg(grp: pd.DataFrame) -> pd.Series:
        tox = grp["toxic_ticks"].dropna().to_numpy()
        rep = grp["reposition_ticks"].dropna().to_numpy()
        tot = grp["total_ticks"].dropna().to_numpy()
        asc = grp["as_score_ticks"].dropna().to_numpy()
        ct  = grp["reposition_case"].value_counts().to_dict()
        n   = len(grp)
        nr  = int(grp["repositioned"].sum())

        mean_tox = float(tox.mean()) if len(tox) else np.nan
        mean_rep = float(rep.mean()) if len(rep) else np.nan
        mean_asc = float(asc.mean()) if len(asc) else np.nan
        mean_tot = float(tot.mean()) if len(tot) else np.nan

        if n < 10:
            verdict = "INSUFFICIENT DATA"
        elif not np.isnan(mean_tox) and mean_tox < -0.5:
            verdict = "ADVERSE SELECTION — toxic flow dominant (mean < -0.5 ticks)"
        elif not np.isnan(mean_tox) and mean_tox > 0:
            verdict = "CLEAN FILLS — spread capture intact"
        elif not np.isnan(mean_rep) and mean_rep < -0.3:
            verdict = "PRIORITY LOSS — repositioning hurts queue position"
        elif not np.isnan(mean_tot) and mean_tot > 0:
            verdict = "MARGINAL POSITIVE"
        else:
            verdict = "BREAK-EVEN / WATCH"

        return pd.Series({
            "n_fills":               n,
            "n_repositioned":        nr,
            "pct_repositioned":      round(nr / max(n, 1) * 100, 1),
            "mean_toxic_ticks":      round(mean_tox, 6)                        if not np.isnan(mean_tox) else np.nan,
            "std_toxic_ticks":       round(float(tox.std()), 6)                if len(tox) > 1           else np.nan,
            "pct_toxic_negative":    round(float((tox < 0).mean() * 100), 1)  if len(tox)               else np.nan,
            "mean_reposition_ticks": round(mean_rep, 6)                        if not np.isnan(mean_rep) else np.nan,
            "pct_reposition_loss":   round(float((rep < 0).mean() * 100), 1)  if len(rep)               else np.nan,
            "mean_as_score_ticks":   round(mean_asc, 6)                        if not np.isnan(mean_asc) else np.nan,
            "pct_as_negative":       round(float((asc < 0).mean() * 100), 1)  if len(asc)               else np.nan,
            "mean_total_ticks":      round(mean_tot, 6)                        if not np.isnan(mean_tot) else np.nan,
            "case_A_pct":            round(ct.get("A_REPOSITION_SAVED_FILL",         0) / max(n,1)*100, 1),
            "case_B_pct":            round(ct.get("B_BOTH_FILLED_PRIORITY_LOSS",      0) / max(n,1)*100, 1),
            "case_C_pct":            round(ct.get("C_SHADOW_FILLED_STANDARD_DID_NOT", 0) / max(n,1)*100, 1),
            "case_D_pct":            round(ct.get("D_NO_REPOSITION",                  0) / max(n,1)*100, 1),
            "verdict":               verdict,
        })

    summary = (
        df.groupby(["level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["level_idx", "side"])
    )

    logger.info(
        "\n"
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║  CONTRIBUTION TABLE — grouped by LOB Level                   ║\n"
        "║  All values in TICKS  (normalised: order_size × tick_size)   ║\n"
        "╚═══════════════════════════════════════════════════════════════╝\n"
        "  toxic_ticks      < 0 → adverse selection\n"
        "  reposition_ticks < 0 → priority loss from repositioning\n"
        "  as_score_ticks   < 0 → structural adverse selection (shadow)\n\n"
        "%s",
        summary[[
            "level_idx", "side", "n_fills",
            "mean_toxic_ticks", "mean_reposition_ticks",
            "mean_as_score_ticks", "mean_total_ticks", "verdict",
        ]].to_string(index=False),
    )
    return summary


def build_date_level_summary(contribution_rows: List[dict]) -> pd.DataFrame:
    """Secondary breakdown: date x level_idx x side."""
    if not contribution_rows:
        return pd.DataFrame()
    df = pd.DataFrame(contribution_rows)

    def _agg(grp):
        tox = grp["toxic_ticks"].dropna().to_numpy()
        tot = grp["total_ticks"].dropna().to_numpy()
        asc = grp["as_score_ticks"].dropna().to_numpy()
        return pd.Series({
            "n_fills":             len(grp),
            "mean_toxic_ticks":    round(float(tox.mean()), 6)            if len(tox) else np.nan,
            "pct_toxic_negative":  round(float((tox < 0).mean() * 100), 1) if len(tox) else np.nan,
            "mean_as_score_ticks": round(float(asc.mean()), 6)            if len(asc) else np.nan,
            "mean_total_ticks":    round(float(tot.mean()), 6)            if len(tot) else np.nan,
            "cumulative_ticks":    round(float(tot.sum()),  4)            if len(tot) else 0.0,
        })

    return (
        df.groupby(["date", "level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["date", "level_idx", "side"])
    )


def run_fqi_analysis(
    contribution_rows: List[dict],
    outdir,
    save_prefix: str,
) -> pd.DataFrame:
    """
    Run full reporting pipeline and save CSVs.
    Replaces WalkForwardValidator.run().
    Returns level-summary DataFrame.

    Saves:
        <prefix>_level_summary.csv
        <prefix>_date_level_summary.csv
        <prefix>_contributions_raw.csv
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if not contribution_rows:
        logger.warning("run_fqi_analysis: no contribution rows.")
        return pd.DataFrame()

    level_df = build_level_summary(contribution_rows)
    level_df.to_csv(out / f"{save_prefix}_level_summary.csv", index=False)
    logger.info("Saved → %s", out / f"{save_prefix}_level_summary.csv")

    date_df = build_date_level_summary(contribution_rows)
    date_df.to_csv(out / f"{save_prefix}_date_level_summary.csv", index=False)
    logger.info("Saved → %s", out / f"{save_prefix}_date_level_summary.csv")

    raw_df   = pd.DataFrame(contribution_rows)
    raw_save = raw_df.drop(columns=["state"], errors="ignore")
    raw_save.to_csv(out / f"{save_prefix}_contributions_raw.csv", index=False)
    logger.info("Saved raw contributions (%d rows) → %s",
                len(raw_df), out / f"{save_prefix}_contributions_raw.csv")

    # Headline log
    tox = raw_df["toxic_ticks"].dropna().to_numpy()
    rep = raw_df["reposition_ticks"].dropna().to_numpy()
    asc = raw_df["as_score_ticks"].dropna().to_numpy()
    tot = raw_df["total_ticks"].dropna().to_numpy()
    cases = raw_df["reposition_case"].value_counts().to_dict()
    n_rep = int(raw_df["repositioned"].sum())

    logger.info(
        "\n"
        "══════════════════════════════════════════════════════════════\n"
        " HEADLINE RESULTS\n"
        "══════════════════════════════════════════════════════════════\n"
        " Fills: %d  |  Repositioned: %d (%.1f%%)\n"
        " Toxic Flow   mean: %+.4f ticks  pct_negative: %.1f%%\n"
        " Reposition   mean: %+.4f ticks\n"
        "   Case A (saved fill): %d  Case B (priority loss): %d\n"
        "   Case C (shadow only): %d  Case D (no reposition): %d\n"
        " AS Score     mean: %+.4f ticks\n"
        " Total        mean: %+.4f ticks  cumul: %+.2f ticks\n"
        "══════════════════════════════════════════════════════════════",
        len(raw_df), n_rep, n_rep / max(len(raw_df), 1) * 100,
        float(tox.mean()) if len(tox) else np.nan,
        float((tox < 0).mean() * 100) if len(tox) else 0.0,
        float(rep.mean()) if len(rep) else np.nan,
        cases.get("A_REPOSITION_SAVED_FILL",         0),
        cases.get("B_BOTH_FILLED_PRIORITY_LOSS",      0),
        cases.get("C_SHADOW_FILLED_STANDARD_DID_NOT", 0),
        cases.get("D_NO_REPOSITION",                  0),
        float(asc.mean()) if len(asc) else np.nan,
        float(tot.mean()) if len(tot) else np.nan,
        float(tot.sum())  if len(tot) else 0.0,
    )

    return level_df



# ─────────────────────────────────────────────────────────────────────────────
# Cancel Value Analysis
# ─────────────────────────────────────────────────────────────────────────────

def cancel_value_analysis(
    cancel_rows: List[dict],
    outdir,
    save_prefix: str,
) -> pd.DataFrame:
    """
    Calculate the "Value of Cancel" (Option Value) for every strand-cancelled
    standard order, then aggregate by LOB level.

    Formula (per cancelled order)
    ------------------------------
        V_cancel = RL_PnL - Shadow_PnL

    RL_PnL (the realised cost at cancel time)
    -----------------------------------------
    The local 100 ms drift penalty at the moment of cancellation.
    This is the mark-to-market loss already "locked in" just before the
    cancel fires.  Normalised to ticks:

        BUY:   rl_pnl_ticks = (mid_at_cancel - birth_mid) / normaliser
               (negative if mid fell since placement — market moved against you)
        SELL:  rl_pnl_ticks = (birth_mid - mid_at_cancel) / normaliser

    If mid_at_cancel is unavailable (NaN), rl_pnl_ticks = 0.

    Shadow_PnL (what would have happened without the cancel)
    --------------------------------------------------------
        Shadow NEVER fills  →  shadow_pnl_ticks = 0  (neutral; order expires worthless)
        Shadow DOES fill    →  Spread Capture + Markout, normalised to ticks:
            BUY:   shadow_pnl_ticks = (shadow_mid_post_100ms - shadow_fill_price) / normaliser
            SELL:  shadow_pnl_ticks = (shadow_fill_price - shadow_mid_post_100ms) / normaliser
            (positive = clean fill, negative = toxic fill even without cancel)

    V_cancel interpretation
    -----------------------
        > 0  JUSTIFIED CANCEL:   you avoided a toxic fill.  The shadow would
                                  have filled at a loss greater than RL_PnL.
        = 0  NEUTRAL:            shadow never filled; cancel had no opportunity
                                  cost but no protection value either.
        < 0  OVER-CANCEL:        you cancelled an order that would have filled
                                  cleanly; you left spread-capture on the table.

    Output columns (one row per cancel)
    ------------------------------------
        virtual_id, date, session_id, side, level_idx
        cancel_mid, birth_mid, mid_100ms_ago
        rl_pnl_ticks          local drift penalty at cancel (RL_PnL)
        shadow_filled         True/False
        shadow_fill_price     NaN if shadow never filled
        shadow_mid_post_100ms NaN if shadow never filled
        shadow_pnl_ticks      0 if shadow never filled
        v_cancel_ticks        = rl_pnl_ticks - shadow_pnl_ticks
        order_size, normaliser

    Level summary columns
    ---------------------
        n_cancels
        n_shadow_filled       cancels where shadow eventually filled
        pct_shadow_filled     n_shadow_filled / n_cancels × 100
        mean_rl_pnl_ticks     average drift penalty at cancel
        mean_shadow_pnl_ticks average shadow PnL (0 for non-fills)
        mean_v_cancel_ticks   average Value of Cancel  ← primary metric
        pct_justified         % where v_cancel > 0  (cancel saved money)
        pct_opportunity_cost  % where v_cancel < 0  (cancel cost money)
        cumulative_v_cancel   sum of v_cancel_ticks across all cancels
        verdict
    """
    if not cancel_rows:
        logger.warning("cancel_value_analysis: no cancel rows.")
        return pd.DataFrame()

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(cancel_rows)

    # ── Raw per-cancel CSV ────────────────────────────────────────────────────
    raw_path = out / f"{save_prefix}_cancel_value_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Cancel value raw (%d rows) → %s", len(df), raw_path)

    # ── Level summary ─────────────────────────────────────────────────────────
    def _agg(grp: pd.DataFrame) -> pd.Series:
        rl   = grp["rl_pnl_ticks"].dropna().to_numpy()
        shd  = grp["shadow_pnl_ticks"].dropna().to_numpy()
        vc   = grp["v_cancel_ticks"].dropna().to_numpy()
        n    = len(grp)
        nsf  = int(grp["shadow_filled"].sum())

        mean_vc = float(vc.mean()) if len(vc) else np.nan

        if n < 5:
            verdict = "INSUFFICIENT DATA"
        elif not np.isnan(mean_vc) and mean_vc > 0.1:
            verdict = "JUSTIFIED — cancels avoiding toxic fills"
        elif not np.isnan(mean_vc) and mean_vc < -0.1:
            verdict = "OVER-CANCELLING — leaving clean fills on the table"
        else:
            verdict = "NEUTRAL — mostly cancels on orders that never would have filled"

        return pd.Series({
            "n_cancels":             n,
            "n_shadow_filled":       nsf,
            "pct_shadow_filled":     round(nsf / max(n, 1) * 100, 1),
            "mean_rl_pnl_ticks":     round(float(rl.mean()),  6) if len(rl)  else np.nan,
            "mean_shadow_pnl_ticks": round(float(shd.mean()), 6) if len(shd) else np.nan,
            "mean_v_cancel_ticks":   round(mean_vc,           6) if not np.isnan(mean_vc) else np.nan,
            "pct_justified":         round(float((vc > 0).mean() * 100), 1) if len(vc) else np.nan,
            "pct_opportunity_cost":  round(float((vc < 0).mean() * 100), 1) if len(vc) else np.nan,
            "cumulative_v_cancel":   round(float(vc.sum()), 4) if len(vc) else 0.0,
            "verdict":               verdict,
        })

    level_df = (
        df.groupby(["level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["level_idx", "side"])
    )

    level_path = out / f"{save_prefix}_cancel_value_by_level.csv"
    level_df.to_csv(level_path, index=False)

    # ── Headline log ──────────────────────────────────────────────────────────
    vc_all  = df["v_cancel_ticks"].dropna().to_numpy()
    nsf_all = int(df["shadow_filled"].sum())
    n_all   = len(df)

    logger.info(
        "\n"
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║  VALUE OF CANCEL ANALYSIS                                    ║\n"
        "║  V_cancel = RL_PnL - Shadow_PnL   (ticks, normalised)       ║\n"
        "╚═══════════════════════════════════════════════════════════════╝\n"
        "  Formula:\n"
        "    RL_PnL      = local drift at cancel (mid_cancel - birth_mid) / norm\n"
        "    Shadow_PnL  = fill value if shadow filled, else 0\n"
        "    V_cancel    = RL_PnL - Shadow_PnL\n"
        "    > 0  Justified (avoided toxic fill)\n"
        "    < 0  Opportunity cost (left clean fill on the table)\n\n"
        "  Population:\n"
        "    Total cancels          : %d\n"
        "    Shadow eventually filled: %d  (%.1f%%)\n"
        "    Shadow never filled    : %d  (%.1f%%)\n\n"
        "  Value of Cancel:\n"
        "    Mean V_cancel          : %+.4f ticks\n"
        "    Pct justified (> 0)    : %.1f%%\n"
        "    Pct opportunity cost (<0): %.1f%%\n"
        "    Cumulative V_cancel    : %+.2f ticks\n\n"
        "  By level:\n%s\n"
        "  Saved → %s",
        n_all,
        nsf_all, nsf_all / max(n_all, 1) * 100,
        n_all - nsf_all, (n_all - nsf_all) / max(n_all, 1) * 100,
        float(vc_all.mean())            if len(vc_all) else np.nan,
        float((vc_all > 0).mean() * 100) if len(vc_all) else 0.0,
        float((vc_all < 0).mean() * 100) if len(vc_all) else 0.0,
        float(vc_all.sum())             if len(vc_all) else 0.0,
        level_df[[
            "level_idx", "side", "n_cancels", "pct_shadow_filled",
            "mean_v_cancel_ticks", "pct_justified", "pct_opportunity_cost", "verdict",
        ]].to_string(index=False),
        level_path,
    )

    return level_df

# ── Backward-compat shims ─────────────────────────────────────────────────────

def build_transition_df(transitions: list) -> pd.DataFrame:
    rows = [
        {"virtual_id": t.get("virtual_id",""), "trigger": t.get("trigger",""),
         "reward": float(t.get("reward", 0.0)), "session_id": int(t.get("session_id",0)),
         "level_idx": int(t.get("level_idx",0)), "side": t.get("side","")}
        for t in transitions if t.get("terminal")
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class WalkForwardValidator:
    """Shim — FQI disabled. Use run_fqi_analysis() instead."""
    def __init__(self, *args, **kwargs):
        logger.info("WalkForwardValidator: FQI is DISABLED. Use run_fqi_analysis().")
        self.best_model_ = None

    def run(self, *args, **kwargs):
        return {}

    def summary_df(self):
        return pd.DataFrame()

    def apply_policy(self, df):
        return df


def option_value_analysis(*args, **kwargs):
    logger.info("option_value_analysis: no-op — use ContributionEngine.calculate().")