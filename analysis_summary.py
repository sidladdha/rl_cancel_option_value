"""
analysis_summary.py
===================
Counterfactual contribution summary — replaces fqi_model.py

No XGBoost, no Bellman updates, no walk-forward training.

Takes the contribution rows from ContributionEngine and produces:
  1. Summary by Level × Side
  2. Summary by Date × Level × Side
  3. Toxic Flow distribution (mean, std, pct negative)
  4. Reposition case breakdown

All output is pure pandas aggregation — runs in < 1 second.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_contribution_df(contribution_rows: List[dict]) -> pd.DataFrame:
    """Convert list of contribution dicts to a DataFrame."""
    df = pd.DataFrame(contribution_rows)
    if df.empty:
        raise ValueError("build_contribution_df: no contribution rows found. "
                         "Check that fills are being detected.")
    logger.info("Contribution DataFrame: %d fills, %d unique dates, %d levels",
                len(df), df["date"].nunique(), df["level_idx"].nunique())
    return df


def summarise_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean contributions grouped by level_idx × side.

    Columns:
        n_fills               : number of filled orders
        n_repositioned        : how many were repositioned before fill
        mean_toxic            : mean toxic flow contribution per fill
        pct_toxic_negative    : % of fills with toxic_contribution < 0
        mean_reposition       : mean reposition contribution
        pct_reposition_loss   : % of repositioned fills with reposition < 0
        mean_total            : mean total contribution
        case_A_pct, B_pct, D_pct  : reposition case breakdown
    """
    def _agg(grp):
        tox = grp["toxic_contribution"].dropna().to_numpy()
        rep = grp["reposition_contribution"].dropna().to_numpy()
        tot = grp["total_contribution"].dropna().to_numpy()
        ct  = grp["reposition_case"].value_counts().to_dict()
        n   = len(grp)
        nr  = int(grp["repositioned"].sum())

        return pd.Series({
            "n_fills":             n,
            "n_repositioned":      nr,
            "pct_repositioned":    round(nr / max(n, 1) * 100, 1),
            "mean_toxic":          round(float(tox.mean()), 6) if len(tox) else np.nan,
            "std_toxic":           round(float(tox.std()),  6) if len(tox) > 1 else np.nan,
            "pct_toxic_negative":  round(float((tox < 0).mean() * 100), 1) if len(tox) else np.nan,
            "mean_reposition":     round(float(rep.mean()), 6) if len(rep) else np.nan,
            "pct_reposition_loss": round(float((rep < 0).mean() * 100), 1) if len(rep) else np.nan,
            "mean_total":          round(float(tot.mean()), 6) if len(tot) else np.nan,
            "case_A_pct":          round(ct.get("A_REPOSITION_SAVED_FILL",      0) / max(n,1)*100, 1),
            "case_B_pct":          round(ct.get("B_BOTH_FILLED_PRIORITY_LOSS",  0) / max(n,1)*100, 1),
            "case_D_pct":          round(ct.get("D_NO_REPOSITION",              0) / max(n,1)*100, 1),
            "verdict": (
                "TOXIC FLOW DOMINANT — fills mostly adverse"
                if (len(tox) > 0 and (tox < 0).mean() > 0.55)
                else "CLEAN FILLS — spread capture intact"
            ),
        })

    summary = (
        df.groupby(["level_idx", "side"], sort=True)
        .apply(_agg)
        .reset_index()
        .sort_values(["level_idx", "side"])
    )

    logger.info(
        "\n=== Contribution Summary by Level × Side ===\n%s",
        summary.to_string(index=False),
    )
    return summary


def summarise_by_date_level(df: pd.DataFrame) -> pd.DataFrame:
    """Full breakdown: date × level_idx × side."""
    def _agg(grp):
        tox = grp["toxic_contribution"].dropna().to_numpy()
        tot = grp["total_contribution"].dropna().to_numpy()
        n   = len(grp)
        return pd.Series({
            "n_fills":            n,
            "mean_toxic":         round(float(tox.mean()), 6) if len(tox) else np.nan,
            "pct_toxic_negative": round(float((tox < 0).mean() * 100), 1) if len(tox) else np.nan,
            "mean_total":         round(float(tot.mean()), 6) if len(tot) else np.nan,
            "total_pnl":          round(float(tot.sum()),  5) if len(tot) else np.nan,
        })

    summary = (
        df.groupby(["date", "level_idx", "side"], sort=True)
        .apply(_agg)
        .reset_index()
        .sort_values(["date", "level_idx", "side"])
    )
    return summary


def run_contribution_analysis(
    contribution_rows: List[dict],
    outdir: str | Path,
    save_prefix: str,
) -> pd.DataFrame:
    """
    Full analysis pipeline. Returns by-level summary DataFrame.
    Saves four CSVs.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if not contribution_rows:
        logger.warning("run_contribution_analysis: no contribution rows.")
        return pd.DataFrame()

    df = build_contribution_df(contribution_rows)

    # ── 1. By level × side ───────────────────────────────────────────────────
    by_level = summarise_by_level(df)
    by_level.to_csv(out / f"{save_prefix}_contribution_by_level.csv", index=False)
    logger.info("Saved contribution_by_level → %s",
                out / f"{save_prefix}_contribution_by_level.csv")

    # ── 2. By date × level × side ─────────────────────────────────────────────
    by_date = summarise_by_date_level(df)
    by_date.to_csv(out / f"{save_prefix}_contribution_by_date_level.csv", index=False)
    logger.info("Saved contribution_by_date_level")

    # ── 3. Raw fill rows ──────────────────────────────────────────────────────
    df.to_csv(out / f"{save_prefix}_contributions_raw.csv", index=False)
    logger.info("Saved contributions_raw (%d rows)", len(df))

    # ── 4. Overall summary ────────────────────────────────────────────────────
    tox_all = df["toxic_contribution"].dropna().to_numpy()
    rep_all = df["reposition_contribution"].dropna().to_numpy()
    tot_all = df["total_contribution"].dropna().to_numpy()
    cases   = df["reposition_case"].value_counts().to_dict()
    n_rep   = int(df["repositioned"].sum())

    logger.info(
        "\n"
        "=== COUNTERFACTUAL CONTRIBUTION ANALYSIS — FULL SUMMARY ===\n\n"
        "Fill population:\n"
        "  Total fills analysed       : %d\n"
        "  Repositioned (10-tick rule): %d (%.1f%%)\n\n"
        "Toxic Flow Contribution (fill_price - mid_T+100ms):\n"
        "  Mean                       : %+.6f  (negative = adverse selection)\n"
        "  Std                        : %.6f\n"
        "  Pct negative (toxic fills) : %.1f%%\n\n"
        "Reposition Contribution:\n"
        "  Mean                       : %+.6f  (negative = priority loss)\n"
        "  Case A — Reposition saved fill      : %d (%.1f%%)\n"
        "  Case B — Both filled, priority loss : %d (%.1f%%)\n"
        "  Case D — Not repositioned           : %d (%.1f%%)\n\n"
        "Total Contribution (toxic + reposition):\n"
        "  Mean                       : %+.6f\n"
        "  Cumulative                 : %+.4f",
        len(df), n_rep, n_rep / max(len(df), 1) * 100,
        float(tox_all.mean()) if len(tox_all) else np.nan,
        float(tox_all.std())  if len(tox_all) > 1 else np.nan,
        float((tox_all < 0).mean() * 100) if len(tox_all) else 0.0,
        float(rep_all.mean()) if len(rep_all) else np.nan,
        cases.get("A_REPOSITION_SAVED_FILL",     0),
        cases.get("A_REPOSITION_SAVED_FILL",     0) / max(len(df), 1) * 100,
        cases.get("B_BOTH_FILLED_PRIORITY_LOSS", 0),
        cases.get("B_BOTH_FILLED_PRIORITY_LOSS", 0) / max(len(df), 1) * 100,
        cases.get("D_NO_REPOSITION",             0),
        cases.get("D_NO_REPOSITION",             0) / max(len(df), 1) * 100,
        float(tot_all.mean()) if len(tot_all) else np.nan,
        float(tot_all.sum())  if len(tot_all) else 0.0,
    )

    return by_level


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat shims so existing `from fqi_model import ...` doesn't crash
# ─────────────────────────────────────────────────────────────────────────────

STATE_COLS     = ["q_frac", "ln_shares_ahead", "OFI", "spread", "ln_order_size"]
ACTION_HOLD    = 0
ACTION_CANCEL  = 1
ACTIONS        = [ACTION_HOLD, ACTION_CANCEL]
GAMMA_DEFAULT  = 0.99
N_ITER_DEFAULT = 8


def build_transition_df(transitions):
    """Shim: returns minimal DataFrame for any downstream code that calls this."""
    rows = []
    for t in transitions:
        if not t.get("terminal"):
            continue
        rows.append({
            "virtual_id": t.get("virtual_id", ""),
            "trigger":    t.get("trigger", ""),
            "reward":     float(t.get("reward", 0.0)),
            "session_id": int(t.get("session_id", 0)),
            "level_idx":  int(t.get("level_idx", 0)),
            "side":       t.get("side", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class WalkForwardValidator:
    """Shim: FQI training is disabled. Logs a clear message."""
    def __init__(self, *args, **kwargs):
        logger.info("WalkForwardValidator: FQI training is DISABLED. "
                    "Running counterfactual contribution analysis only.")

    def run(self, *args, **kwargs):
        logger.info("WalkForwardValidator.run: no-op — FQI disabled.")
        return {}
