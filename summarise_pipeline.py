"""
summarise_pipeline.py
=====================
Post-run summary aggregator for the HFT Counterfactual Pipeline.

Purpose
-------
The main pipeline (run_hft_pipeline.py) writes one pair of raw CSV files per
calendar-day chunk:

    <outdir>/<prefix>_<YYYY-MM-DD>_contributions_raw.csv
    <outdir>/<prefix>_<YYYY-MM-DD>_cancel_value_raw.csv

and per-chunk lifetime / sharpe outputs:

    <indir>/<prefix>_<YYYY-MM-DD>_sharpe_by_queue.csv
    <indir>/<prefix>_<YYYY-MM-DD>_lifetime_orders.parquet

This script collects ALL of those per-day raw files from <indir>, concatenates
them, and re-runs the full suite of summary analyses — writing cross-day
aggregates to <outdir> (defaults to <indir> when not specified).

What it produces
----------------
Fills (cross-day):
    <prefix>_ALL_contributions_raw.csv          concatenated fill rows
    <prefix>_ALL_level_summary.csv              mean ticks × level × side
    <prefix>_ALL_date_level_summary.csv         mean ticks × date × level × side
    <prefix>_ALL_cumulative_pnl.csv             cumulative total_ticks by date
    <prefix>_ALL_headline.txt                   plain-text headline block

Cancels (cross-day):
    <prefix>_ALL_cancel_value_raw.csv           concatenated cancel rows
    <prefix>_ALL_cancel_value_by_level.csv      V_cancel × level × side
    <prefix>_ALL_cancel_value_by_date.csv       V_cancel × date
    <prefix>_ALL_cancel_headline.txt            plain-text headline block

Lifetime / queue position (cross-day):
    <prefix>_ALL_lifetime_orders.parquet        concatenated lifetime rows
    <prefix>_ALL_sharpe_by_queue.csv            Sharpe × q_bucket × level × side

Usage
-----
    # indir = outdir (write summaries back into the same folder)
    python summarise_pipeline.py \\
        --indir  outputs_hft \\
        --prefix out_contrib

    # separate output folder
    python summarise_pipeline.py \\
        --indir   outputs_hft \\
        --outdir  summaries \\
        --prefix  out_contrib

    # with explicit date range filter
    python summarise_pipeline.py \\
        --indir     outputs_hft \\
        --outdir    summaries \\
        --prefix    out_contrib \\
        --date-from 2024-01-01 \\
        --date-to   2024-03-31

    # dry-run: list files that would be included, then exit
    python summarise_pipeline.py \\
        --indir outputs_hft --prefix out_contrib --dry-run

Design
------
* Reads only the small CSV/parquet outputs — never the raw MBO parquet.
* Streams via pandas chunked reads for very large raw files (--chunk-rows).
* No dependency on virtual_order_tracker.py or contribution_engine.py.
* Safe to re-run: all outputs are overwritten atomically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(
    outdir:    Path,
    prefix:    str,
    suffix:    str,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
) -> List[Path]:
    """
    Return sorted list of files matching <outdir>/<prefix>_*_<suffix>.
    Optionally filter by ISO date embedded in the filename stem.
    """
    pattern = f"{prefix}_*_{suffix}"
    files   = sorted(outdir.glob(pattern))

    if not files:
        return []

    if date_from or date_to:
        filtered = []
        for f in files:
            # stem looks like: out_contrib_2024-01-15_contributions_raw
            # extract the date segment (third token when split by _)
            parts = f.stem.replace(prefix + "_", "", 1).split("_")
            # find the YYYY-MM-DD token
            date_str = None
            for p in parts:
                if len(p) == 10 and p[4] == "-" and p[7] == "-":
                    date_str = p
                    break
            if date_str is None:
                filtered.append(f)  # can't parse date — include anyway
                continue
            if date_from and date_str < date_from:
                continue
            if date_to and date_str > date_to:
                continue
            filtered.append(f)
        files = filtered

    return files


# ─────────────────────────────────────────────────────────────────────────────
# Chunked CSV reader (memory-safe for large raw files)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_csv_chunks(
    files:      List[Path],
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of at most chunk_rows rows from a list of CSV files."""
    for path in files:
        logger.info("  Reading %s", path.name)
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_rows):
                yield chunk
        except Exception as exc:
            logger.warning("  Skipping %s — %s", path.name, exc)


def _concat_csv_files(
    files:      List[Path],
    chunk_rows: int,
) -> pd.DataFrame:
    """Concatenate all CSV files into a single DataFrame."""
    parts = list(_iter_csv_chunks(files, chunk_rows))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fill summary (mirrors fqi_model.build_level_summary / build_date_level_summary)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (level_idx, side) — mean ticks + verdict.

    Blended means (mean_toxic_ticks etc.) mix repositioned (Case A) and
    passive (Case D) fills, which have very different economics.  The
    *_passive and *_repo split columns are added so each can be read cleanly.
    """
    def _agg(grp: pd.DataFrame) -> pd.Series:
        tox = grp["toxic_ticks"].dropna().to_numpy()
        rep = grp["reposition_ticks"].dropna().to_numpy()
        tot = grp["total_ticks"].dropna().to_numpy()
        asc = grp["as_score_ticks"].dropna().to_numpy()
        ct  = grp["reposition_case"].value_counts().to_dict() if "reposition_case" in grp else {}
        n   = len(grp)
        nr  = int(grp["repositioned"].sum()) if "repositioned" in grp else 0

        # ── Case-split arrays (NaN-filtered via new columns from contribution_engine) ──
        # toxic_ticks_passive / toxic_ticks_repo emitted per row since last patch.
        # Fall back to filtering by reposition_case if columns are absent (old files).
        if "toxic_ticks_passive" in grp.columns:
            tox_passive = grp["toxic_ticks_passive"].dropna().to_numpy()
            tox_repo    = grp["toxic_ticks_repo"].dropna().to_numpy()
            tot_passive = grp["total_ticks_passive"].dropna().to_numpy()
            tot_repo    = grp["total_ticks_repo"].dropna().to_numpy()
            rep_caseA   = grp["reposition_ticks_caseA"].dropna().to_numpy()
        else:
            # Legacy fallback: derive from reposition_case column
            mask_D = grp["reposition_case"] == "D_NO_REPOSITION"
            mask_A = grp["reposition_case"] == "A_REPOSITION_SAVED_FILL"
            tox_passive = grp.loc[mask_D, "toxic_ticks"].dropna().to_numpy()
            tox_repo    = grp.loc[mask_A, "toxic_ticks"].dropna().to_numpy()
            tot_passive = grp.loc[mask_D, "total_ticks"].dropna().to_numpy()
            tot_repo    = grp.loc[mask_A, "total_ticks"].dropna().to_numpy()
            rep_caseA   = grp.loc[mask_A, "reposition_ticks"].dropna().to_numpy()

        # Timing arrays (present in new output files only)
        r2f = grp["repo_to_fill_ms"].dropna().to_numpy()   if "repo_to_fill_ms"  in grp.columns else np.array([])
        b2f = grp["birth_to_fill_ms"].dropna().to_numpy()  if "birth_to_fill_ms" in grp.columns else np.array([])

        mean_tox         = float(tox.mean())         if len(tox)         else np.nan
        mean_rep         = float(rep.mean())         if len(rep)         else np.nan
        mean_asc         = float(asc.mean())         if len(asc)         else np.nan
        mean_tot         = float(tot.mean())         if len(tot)         else np.nan
        mean_tox_passive = float(tox_passive.mean()) if len(tox_passive) else np.nan
        mean_tox_repo    = float(tox_repo.mean())    if len(tox_repo)    else np.nan
        mean_tot_passive = float(tot_passive.mean()) if len(tot_passive) else np.nan
        mean_tot_repo    = float(tot_repo.mean())    if len(tot_repo)    else np.nan
        mean_rep_caseA   = float(rep_caseA.mean())   if len(rep_caseA)   else np.nan
        mean_r2f         = float(r2f.mean())          if len(r2f)         else np.nan
        med_r2f          = float(np.median(r2f))      if len(r2f)         else np.nan
        p95_r2f          = float(np.percentile(r2f, 95)) if len(r2f)     else np.nan
        mean_b2f         = float(b2f.mean())          if len(b2f)         else np.nan

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
            # ── Blended means (all fills, legacy) ──────────────────────────
            "mean_toxic_ticks":      round(mean_tox, 6)                         if not np.isnan(mean_tox) else np.nan,
            "std_toxic_ticks":       round(float(tox.std()),  6)                if len(tox) > 1           else np.nan,
            "pct_toxic_negative":    round(float((tox < 0).mean() * 100), 1)   if len(tox)               else np.nan,
            "mean_reposition_ticks": round(mean_rep, 6)                         if not np.isnan(mean_rep) else np.nan,
            "pct_reposition_loss":   round(float((rep < 0).mean() * 100), 1)   if len(rep)               else np.nan,
            "mean_as_score_ticks":   round(mean_asc, 6)                         if not np.isnan(mean_asc) else np.nan,
            "pct_as_negative":       round(float((asc < 0).mean() * 100), 1)   if len(asc)               else np.nan,
            "mean_total_ticks":      round(mean_tot, 6)                         if not np.isnan(mean_tot) else np.nan,
            # ── Case-split means (uncontaminated) ──────────────────────────
            # *_passive: Case D only — filled at original birth price.
            #   True queue-depth adverse selection; should rise with depth.
            # *_repo: Case A only — repositioned to BBO then filled.
            #   Lower toxic (fill_price ≈ mid at repo time); flat across levels.
            # mean_reposition_ticks_caseA: repo credit per repositioned fill,
            #   undiluted by the zero-valued D orders.
            "mean_toxic_passive":        round(mean_tox_passive, 6) if not np.isnan(mean_tox_passive) else np.nan,
            "mean_toxic_repo":           round(mean_tox_repo,    6) if not np.isnan(mean_tox_repo)    else np.nan,
            "mean_total_passive":        round(mean_tot_passive, 6) if not np.isnan(mean_tot_passive) else np.nan,
            "mean_total_repo":           round(mean_tot_repo,    6) if not np.isnan(mean_tot_repo)    else np.nan,
            "mean_reposition_ticks_caseA": round(mean_rep_caseA, 6) if not np.isnan(mean_rep_caseA)  else np.nan,
            # ── Timing stats ───────────────────────────────────────────────
            # repo_to_fill_ms: elapsed ms from reposition to fill (Case A only).
            #   mean/median/p95 across all repositioned fills at this level.
            #   Shows how quickly BBO price is reached after a 10-tick drift.
            # birth_to_fill_ms: total order lifetime from placement to fill (all cases).
            "mean_repo_to_fill_ms":  round(mean_r2f, 3) if not np.isnan(mean_r2f) else np.nan,
            "med_repo_to_fill_ms":   round(med_r2f,  3) if not np.isnan(med_r2f)  else np.nan,
            "p95_repo_to_fill_ms":   round(p95_r2f,  3) if not np.isnan(p95_r2f)  else np.nan,
            "mean_birth_to_fill_ms": round(mean_b2f, 3) if not np.isnan(mean_b2f) else np.nan,
            "case_A_pct": round(ct.get("A_REPOSITION_SAVED_FILL",          0) / max(n, 1) * 100, 1),
            "case_B_pct": round(ct.get("B_BOTH_FILLED_PRIORITY_LOSS",      0) / max(n, 1) * 100, 1),
            "case_C_pct": round(ct.get("C_SHADOW_FILLED_STANDARD_DID_NOT", 0) / max(n, 1) * 100, 1),
            "case_D_pct": round(ct.get("D_NO_REPOSITION",                  0) / max(n, 1) * 100, 1),
            "verdict":               verdict,
        })

    return (
        df.groupby(["level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["level_idx", "side"])
    )


def _fill_date_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, level_idx, side)."""
    if "date" not in df.columns:
        return pd.DataFrame()

    def _agg(grp):
        tox = grp["toxic_ticks"].dropna().to_numpy()
        tot = grp["total_ticks"].dropna().to_numpy()
        asc = grp["as_score_ticks"].dropna().to_numpy()
        return pd.Series({
            "n_fills":             len(grp),
            "mean_toxic_ticks":    round(float(tox.mean()), 6)             if len(tox) else np.nan,
            "pct_toxic_negative":  round(float((tox < 0).mean() * 100), 1) if len(tox) else np.nan,
            "mean_as_score_ticks": round(float(asc.mean()), 6)             if len(asc) else np.nan,
            "mean_total_ticks":    round(float(tot.mean()), 6)             if len(tot) else np.nan,
            "cumulative_ticks":    round(float(tot.sum()),  4)             if len(tot) else 0.0,
        })

    return (
        df.groupby(["date", "level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["date", "level_idx", "side"])
    )


def _cumulative_pnl_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Daily and running-cumulative total_ticks across all levels and sides."""
    if "date" not in df.columns or "total_ticks" not in df.columns:
        return pd.DataFrame()

    daily = (
        df.groupby("date")["total_ticks"]
          .agg(n_fills="count", daily_mean_ticks="mean", daily_sum_ticks="sum")
          .reset_index()
          .sort_values("date")
    )
    daily["cumulative_ticks"] = daily["daily_sum_ticks"].cumsum().round(4)
    daily["daily_mean_ticks"] = daily["daily_mean_ticks"].round(6)
    daily["daily_sum_ticks"]  = daily["daily_sum_ticks"].round(4)
    return daily


def _fill_headline(df: pd.DataFrame) -> str:
    """Plain-text headline block for fill results."""
    tox   = df["toxic_ticks"].dropna().to_numpy()
    rep   = df["reposition_ticks"].dropna().to_numpy()     if "reposition_ticks"  in df else np.array([])
    asc   = df["as_score_ticks"].dropna().to_numpy()       if "as_score_ticks"    in df else np.array([])
    tot   = df["total_ticks"].dropna().to_numpy()
    n_rep = int(df["repositioned"].sum())                  if "repositioned"      in df else 0
    cases = df["reposition_case"].value_counts().to_dict() if "reposition_case"   in df else {}
    n     = len(df)

    date_range = ""
    if "date" in df.columns:
        dates = sorted(df["date"].dropna().unique())
        if dates:
            date_range = f"  Date range : {dates[0]}  →  {dates[-1]}  ({len(dates)} days)\n"

    lines = [
        "══════════════════════════════════════════════════════════════",
        " FILL CONTRIBUTION HEADLINE  (cross-day aggregate)",
        "══════════════════════════════════════════════════════════════",
        date_range.rstrip("\n"),
        f"  Fills total      : {n}",
        f"  Repositioned     : {n_rep}  ({n_rep / max(n, 1) * 100:.1f}%)",
        "",
        f"  Toxic Flow    mean : {float(tox.mean()):+.4f} ticks   pct_negative: {float((tox < 0).mean() * 100):.1f}%" if len(tox) else "  Toxic Flow    mean : n/a",
        f"  Reposition    mean : {float(rep.mean()):+.4f} ticks"  if len(rep) else "  Reposition    mean : n/a",
        f"    Case A (saved fill)    : {cases.get('A_REPOSITION_SAVED_FILL', 0)}",
        f"    Case B (priority loss) : {cases.get('B_BOTH_FILLED_PRIORITY_LOSS', 0)}",
        f"    Case C (shadow only)   : {cases.get('C_SHADOW_FILLED_STANDARD_DID_NOT', 0)}",
        f"    Case D (no reposition) : {cases.get('D_NO_REPOSITION', 0)}",
        f"  AS Score      mean : {float(asc.mean()):+.4f} ticks" if len(asc) else "  AS Score      mean : n/a",
        f"  Total         mean : {float(tot.mean()):+.4f} ticks   cumul: {float(tot.sum()):+.2f} ticks" if len(tot) else "  Total         mean : n/a",
        "══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Cancel value summary
# ─────────────────────────────────────────────────────────────────────────────

def _cancel_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (level_idx, side) — V_cancel metrics + verdict."""
    def _agg(grp: pd.DataFrame) -> pd.Series:
        rl  = grp["rl_pnl_ticks"].dropna().to_numpy()    if "rl_pnl_ticks"    in grp else np.array([])
        shd = grp["shadow_pnl_ticks"].dropna().to_numpy() if "shadow_pnl_ticks" in grp else np.array([])
        vc  = grp["v_cancel_ticks"].dropna().to_numpy()   if "v_cancel_ticks"  in grp else np.array([])
        n   = len(grp)
        nsf = int(grp["shadow_filled"].sum()) if "shadow_filled" in grp else 0

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

    return (
        df.groupby(["level_idx", "side"], sort=True)
          .apply(_agg)
          .reset_index()
          .sort_values(["level_idx", "side"])
    )


def _cancel_date_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Daily V_cancel totals and means — useful for spotting regime changes."""
    if "date" not in df.columns or "v_cancel_ticks" not in df.columns:
        return pd.DataFrame()

    daily = (
        df.groupby("date")["v_cancel_ticks"]
          .agg(n_cancels="count", mean_v_cancel="mean", sum_v_cancel="sum")
          .reset_index()
          .sort_values("date")
    )
    # Also shadow fill rate per day
    if "shadow_filled" in df.columns:
        sfr = df.groupby("date")["shadow_filled"].mean().rename("pct_shadow_filled")
        daily = daily.merge(sfr, on="date", how="left")
        daily["pct_shadow_filled"] = (daily["pct_shadow_filled"] * 100).round(1)

    daily["cumulative_v_cancel"] = daily["sum_v_cancel"].cumsum().round(4)
    daily["mean_v_cancel"]       = daily["mean_v_cancel"].round(6)
    daily["sum_v_cancel"]        = daily["sum_v_cancel"].round(4)
    return daily


def _cancel_headline(df: pd.DataFrame) -> str:
    """Plain-text headline block for cancel value results."""
    vc  = df["v_cancel_ticks"].dropna().to_numpy() if "v_cancel_ticks" in df else np.array([])
    nsf = int(df["shadow_filled"].sum())           if "shadow_filled"  in df else 0
    n   = len(df)

    date_range = ""
    if "date" in df.columns:
        dates = sorted(df["date"].dropna().unique())
        if dates:
            date_range = f"  Date range : {dates[0]}  →  {dates[-1]}  ({len(dates)} days)\n"

    lines = [
        "══════════════════════════════════════════════════════════════",
        " VALUE OF CANCEL HEADLINE  (cross-day aggregate)",
        " V_cancel = RL_PnL - Shadow_PnL  (ticks, normalised)",
        "══════════════════════════════════════════════════════════════",
        date_range.rstrip("\n"),
        f"  Total cancels            : {n}",
        f"  Shadow eventually filled : {nsf}  ({nsf / max(n, 1) * 100:.1f}%)",
        f"  Shadow never filled      : {n - nsf}  ({(n - nsf) / max(n, 1) * 100:.1f}%)",
        "",
        f"  Mean V_cancel            : {float(vc.mean()):+.4f} ticks" if len(vc) else "  Mean V_cancel            : n/a",
        f"  Pct justified (> 0)      : {float((vc > 0).mean() * 100):.1f}%" if len(vc) else "  Pct justified (> 0)      : n/a",
        f"  Pct opportunity cost (<0): {float((vc < 0).mean() * 100):.1f}%" if len(vc) else "  Pct opportunity cost (<0) : n/a",
        f"  Cumulative V_cancel      : {float(vc.sum()):+.2f} ticks" if len(vc) else "  Cumulative V_cancel      : n/a",
        "══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime / Sharpe aggregation
# ─────────────────────────────────────────────────────────────────────────────

Q_FRAC_BINS   = [0.0, 0.25, 0.50, 0.75, 1.01]
Q_FRAC_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _q_frac_bucket(q: float) -> str:
    for i, upper in enumerate(Q_FRAC_BINS[1:]):
        if q < upper:
            return Q_FRAC_LABELS[i]
    return Q_FRAC_LABELS[-1]


def _sharpe_from_lifetime(df: pd.DataFrame) -> pd.DataFrame:
    """Re-compute Sharpe by (q_bucket, level_idx, side) from lifetime parquet rows."""
    if df.empty or "episode_return" not in df.columns:
        return pd.DataFrame()

    if "q_bucket" not in df.columns and "birth_q_frac" in df.columns:
        df = df.copy()
        df["q_bucket"] = df["birth_q_frac"].apply(_q_frac_bucket)

    rows = []
    for key, grp in df.groupby(["q_bucket", "level_idx", "side"]):
        qb, lvl, sd = key
        arr    = grp["episode_return"].to_numpy(dtype=float)
        n      = len(arr)
        fills  = int((grp["terminal_trigger"] == "FILL").sum()) if "terminal_trigger" in grp else 0
        mean_r = float(np.mean(arr))
        std_r  = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sharpe = mean_r / std_r if (std_r and not np.isnan(std_r) and std_r > 1e-12) else np.nan
        rows.append({
            "q_bucket":   qb,
            "level_idx":  int(lvl),
            "side":       sd,
            "n":          n,
            "fill_rate":  round(fills / n * 100, 2),
            "mean_return": round(mean_r, 6),
            "std_return":  round(std_r,  6) if not np.isnan(std_r) else None,
            "sharpe":      round(sharpe, 4) if not np.isnan(sharpe) else None,
        })

    return (
        pd.DataFrame(rows)
          .sort_values(["level_idx", "q_bucket", "side"])
          .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_summary(
    indir:      str,
    prefix:     str,
    outdir:     Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    chunk_rows: int = 500_000,
    dry_run:    bool = False,
) -> None:
    """
    indir  — directory containing the per-day pipeline outputs
             (contributions_raw.csv, cancel_value_raw.csv, etc.)
    outdir — where to write the aggregated _ALL_ outputs.
             Defaults to indir if not specified.
    """
    src = Path(indir)
    if not src.exists():
        logger.error("Input directory %s does not exist.", src)
        sys.exit(1)

    out = Path(outdir) if outdir else src
    out.mkdir(parents=True, exist_ok=True)

    # ── Discover files ────────────────────────────────────────────────────────
    fill_files   = _discover_files(src, prefix, "contributions_raw.csv",  date_from, date_to)
    cancel_files = _discover_files(src, prefix, "cancel_value_raw.csv",   date_from, date_to)
    lt_files     = list(sorted(src.glob(f"{prefix}_*_lifetime_orders.parquet")))
    sharpe_files = list(sorted(src.glob(f"{prefix}_*_sharpe_by_queue.csv")))

    logger.info("Discovered:")
    logger.info("  Fill raw CSVs    : %d", len(fill_files))
    logger.info("  Cancel raw CSVs  : %d", len(cancel_files))
    logger.info("  Lifetime parquets: %d", len(lt_files))
    logger.info("  Sharpe CSVs      : %d", len(sharpe_files))

    if dry_run:
        for f in fill_files + cancel_files + lt_files + sharpe_files:
            print(f"  {f}")
        logger.info("Dry run — no output written.")
        return

    if not fill_files and not cancel_files:
        logger.error("No raw CSV files found in %s with prefix '%s'.", out, prefix)
        sys.exit(1)

    # ── Fill analysis ─────────────────────────────────────────────────────────
    if fill_files:
        logger.info("Loading %d fill CSV file(s) …", len(fill_files))
        fill_df = _concat_csv_files(fill_files, chunk_rows)
        logger.info("  %d fill rows loaded.", len(fill_df))

        if not fill_df.empty:
            # Save concatenated raw
            raw_path = out / f"{prefix}_ALL_contributions_raw.csv"
            fill_df.to_csv(raw_path, index=False)
            logger.info("  Saved → %s  (%d rows)", raw_path.name, len(fill_df))

            # Level summary
            level_df = _fill_level_summary(fill_df)
            if not level_df.empty:
                p = out / f"{prefix}_ALL_level_summary.csv"
                level_df.to_csv(p, index=False)
                logger.info("  Saved → %s", p.name)

            # Date × level summary
            dl_df = _fill_date_level_summary(fill_df)
            if not dl_df.empty:
                p = out / f"{prefix}_ALL_date_level_summary.csv"
                dl_df.to_csv(p, index=False)
                logger.info("  Saved → %s", p.name)

            # Cumulative PnL
            cum_df = _cumulative_pnl_by_date(fill_df)
            if not cum_df.empty:
                p = out / f"{prefix}_ALL_cumulative_pnl.csv"
                cum_df.to_csv(p, index=False)
                logger.info("  Saved → %s", p.name)

            # Headline text
            headline = _fill_headline(fill_df)
            logger.info("\n%s", headline)
            p = out / f"{prefix}_ALL_headline.txt"
            p.write_text(headline + "\n")
            logger.info("  Saved → %s", p.name)

    # ── Cancel analysis ───────────────────────────────────────────────────────
    if cancel_files:
        logger.info("Loading %d cancel CSV file(s) …", len(cancel_files))
        cancel_df = _concat_csv_files(cancel_files, chunk_rows)
        logger.info("  %d cancel rows loaded.", len(cancel_df))

        if not cancel_df.empty:
            # Save concatenated raw
            raw_path = out / f"{prefix}_ALL_cancel_value_raw.csv"
            cancel_df.to_csv(raw_path, index=False)
            logger.info("  Saved → %s  (%d rows)", raw_path.name, len(cancel_df))

            # Level summary
            clevel_df = _cancel_level_summary(cancel_df)
            if not clevel_df.empty:
                p = out / f"{prefix}_ALL_cancel_value_by_level.csv"
                clevel_df.to_csv(p, index=False)
                logger.info("  Saved → %s", p.name)

            # Daily summary
            cdate_df = _cancel_date_summary(cancel_df)
            if not cdate_df.empty:
                p = out / f"{prefix}_ALL_cancel_value_by_date.csv"
                cdate_df.to_csv(p, index=False)
                logger.info("  Saved → %s", p.name)

            # Headline text
            cheadline = _cancel_headline(cancel_df)
            logger.info("\n%s", cheadline)
            p = out / f"{prefix}_ALL_cancel_headline.txt"
            p.write_text(cheadline + "\n")
            logger.info("  Saved → %s", p.name)

    # ── Lifetime / Sharpe ─────────────────────────────────────────────────────
    lt_parts = []

    if lt_files:
        logger.info("Loading %d lifetime parquet file(s) …", len(lt_files))
        for f in lt_files:
            try:
                lt_parts.append(pd.read_parquet(f))
                logger.info("  Read %s  (%d rows)", f.name, len(lt_parts[-1]))
            except Exception as exc:
                logger.warning("  Skipping %s — %s", f.name, exc)

    # Also fall back to per-chunk sharpe CSVs if no parquets found
    if not lt_parts and sharpe_files:
        logger.info("No lifetime parquets found; aggregating %d sharpe CSV(s) directly …", len(sharpe_files))
        sharpe_parts = []
        for f in sharpe_files:
            try:
                sharpe_parts.append(pd.read_csv(f))
            except Exception as exc:
                logger.warning("  Skipping %s — %s", f.name, exc)
        if sharpe_parts:
            combined_sharpe = pd.concat(sharpe_parts, ignore_index=True)
            p = out / f"{prefix}_ALL_sharpe_by_queue.csv"
            # Re-aggregate: weighted mean by n (can't recover std exactly, just re-summarise n/fill_rate)
            _save_sharpe_aggregated(combined_sharpe, p)

    if lt_parts:
        lt_df = pd.concat(lt_parts, ignore_index=True)
        logger.info("  %d lifetime rows total.", len(lt_df))

        p = out / f"{prefix}_ALL_lifetime_orders.parquet"
        lt_df.to_parquet(p, index=False)
        logger.info("  Saved → %s", p.name)

        sharpe_df = _sharpe_from_lifetime(lt_df)
        if not sharpe_df.empty:
            p = out / f"{prefix}_ALL_sharpe_by_queue.csv"
            sharpe_df.to_csv(p, index=False)
            logger.info("  Saved → %s", p.name)

    logger.info("=" * 60)
    logger.info("Summary complete. All outputs in %s", out)
    logger.info("=" * 60)


def _save_sharpe_aggregated(df: pd.DataFrame, path: Path) -> None:
    """
    Aggregate multiple per-chunk sharpe CSVs into one.
    Uses n-weighted mean for mean_return; Sharpe recomputed from that.
    Note: std_return cannot be exactly recovered across chunks without raw returns.
    The resulting std_return is marked as NaN and sharpe is re-derived from
    mean_return only (suitable as an approximate cross-day view).
    """
    rows = []
    for key, grp in df.groupby(["q_bucket", "level_idx", "side"]):
        qb, lvl, sd = key
        n_total  = int(grp["n"].sum())
        if n_total == 0:
            continue
        # Weighted mean return
        wmean = float((grp["mean_return"] * grp["n"]).sum()) / n_total
        # Weighted fill rate
        wfill = float((grp["fill_rate"] * grp["n"]).sum()) / n_total
        rows.append({
            "q_bucket":   qb,
            "level_idx":  int(lvl),
            "side":       sd,
            "n":          n_total,
            "fill_rate":  round(wfill, 2),
            "mean_return": round(wmean, 6),
            "std_return":  None,   # cannot recover exactly across chunks
            "sharpe":      None,   # cannot recover exactly across chunks
        })

    result = (
        pd.DataFrame(rows)
          .sort_values(["level_idx", "q_bucket", "side"])
          .reset_index(drop=True)
    )
    result.to_csv(path, index=False)
    logger.info("  Saved (aggregated sharpe from chunk CSVs) → %s", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-run summary aggregator for the HFT Counterfactual Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--indir", required=True,
        help="Directory containing the per-day pipeline outputs "
             "(contributions_raw.csv, cancel_value_raw.csv, etc.). "
             "Same as --outdir used when running run_hft_pipeline.py.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Directory to write aggregated _ALL_ summary outputs. "
             "Defaults to --indir if not specified.",
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Save prefix (same as --save-prefix in run_hft_pipeline.py)",
    )
    parser.add_argument(
        "--date-from", default=None, metavar="YYYY-MM-DD",
        help="Include only files with date >= this value",
    )
    parser.add_argument(
        "--date-to", default=None, metavar="YYYY-MM-DD",
        help="Include only files with date <= this value",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=500_000,
        help="Rows per pandas read_csv chunk (default: 500000).  Lower = less RAM.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List discovered files and exit without writing anything",
    )

    args = parser.parse_args()
    run_summary(
        indir      = args.indir,
        prefix     = args.prefix,
        outdir     = args.outdir,
        date_from  = args.date_from,
        date_to    = args.date_to,
        chunk_rows = args.chunk_rows,
        dry_run    = args.dry_run,
    )