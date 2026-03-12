"""
run_hft_pipeline.py
===================
Counterfactual Contribution Analysis Pipeline

Pipeline flow
-------------
  Stage 0  create_multiday_data.py  →  multi_day_data.parquet         [existing]
  Stage 1  Message Processing       →  MBO stream + mid-price array
  Stage 2  Tracker                  →  Standard + Shadow VirtualOrderTracker
  Stage 3  Contribution Calculation →  ContributionEngine (tick-normalised)
             3a  Fill contributions  →  toxic_ticks, reposition_ticks
             3b  Cancel values       →  V_cancel = RL_PnL - Shadow_PnL
  Stage 4  Fill Summary Statistics  →  fqi_model.run_fqi_analysis()
  Stage 5  Run summarise_pipeline.py after all days to aggregate outputs

No FQI training.  No XGBoost.  No Bellman updates.
The pipeline measures what actually happened (standard tracker) vs what
would have happened if no cancel rules fired (shadow tracker) and
quantifies the difference in ticks.

Usage
-----
python run_hft_pipeline.py \\
    --raw-glob       "/data/xnas-itch-*.mbo.csv" \\
    --instrument-id  7152 \\
    --workdir        "." \\
    --outdir         "outputs_hft" \\
    --save-prefix    "out_contrib" \\
    --tick-size      0.01 \\
    --warmup-minutes 10 \\
    --days-per-chunk 1

--days-per-chunk controls memory vs throughput:
    1  = one calendar day per chunk  (lowest memory, safe default)
    5  = five days per chunk         (fewer chunks, more fills per summary)
    0  = not valid; use 1 minimum
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from virtual_order_tracker import (
    VirtualOrderTracker, MODE_STANDARD, MODE_SHADOW,
    TRIG_FILL,
)
from contribution_engine import ContributionEngine
# run_fqi_analysis and cancel_value_analysis are no longer called from the
# main pipeline. Cross-day summaries are produced by summarise_pipeline.py.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: create_multiday_data.py  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def stage0_multiday(raw_glob: str, instrument_id: int, out_path: Path) -> None:
    if out_path.exists():
        logger.info("Stage 0: parquet already exists at %s — skipping.", out_path)
        return
    logger.info("Stage 0: building multi-day parquet …")
    _run([sys.executable, "create_multiday_data.py",
          "--glob", raw_glob,
          "--instrument-id", str(instrument_id),
          "--out", str(out_path)])
    logger.info("Stage 0: done → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Book sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

Q_FRAC_BINS   = [0.0, 0.25, 0.50, 0.75, 1.01]
Q_FRAC_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _q_frac_bucket(q_frac: float) -> str:
    for i, upper in enumerate(Q_FRAC_BINS[1:]):
        if q_frac < upper:
            return Q_FRAC_LABELS[i]
    return Q_FRAC_LABELS[-1]


def _sample_queue_placement(tracker: VirtualOrderTracker, side: str,
                             rng, max_levels: int = 10) -> Optional[tuple]:
    """
    Sample a random placement within the live book.
    Returns (price, total_shares_ahead, shares_in_level, level_idx, level_total)
    or None if the book is not ready.
    """
    book     = tracker._bid_book if side == "B" else tracker._ask_book
    opp_book = tracker._ask_book if side == "B" else tracker._bid_book
    levels   = book.top_n_levels(max_levels)
    if not levels:
        return None

    opp_best = opp_book.best()
    if opp_best is not None:
        levels = [(p, s) for p, s in levels
                  if (p < opp_best if side == "B" else p > opp_best)]
    if not levels:
        return None

    level_idx                = rng.randint(0, len(levels) - 1)
    price, level_total       = levels[level_idx]
    if level_total <= 0:
        return None

    shares_in_level = rng.randint(0, level_total)
    # levels is sorted best-first from top_n_levels.
    # shares strictly ahead of levels[level_idx] = sum of shares in levels[0..level_idx-1]
    # This avoids a second full book scan.
    shares_at_better   = sum(s for _, s in levels[:level_idx])
    total_shares_ahead = shares_at_better + shares_in_level
    return price, total_shares_ahead, shares_in_level, level_idx, level_total


# ─────────────────────────────────────────────────────────────────────────────
# Order Lifetime Tracker  (metadata + Sharpe by queue position)
# ─────────────────────────────────────────────────────────────────────────────

class OrderLifetimeTracker:
    """Records per-order metadata and computes fill-rate / Sharpe by queue bucket."""

    def __init__(self) -> None:
        self._live: Dict[str, dict] = {}
        self.completed: List[dict]  = []

    def record(self, events: List[dict]) -> None:
        from datetime import datetime, timezone
        for t in events:
            vid = t["virtual_id"]
            if vid not in self._live:
                bns = t.get("birth_ts_ns", 0)
                bdate = datetime.fromtimestamp(bns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
                self._live[vid] = {
                    "side":               t.get("side", "?"),
                    "birth_mid":          t.get("birth_mid", np.nan),
                    "birth_ts_ns":        bns,
                    "birth_date":         bdate,
                    "level_idx":          t.get("level_idx", 0),   # 0-based birth level
                    "birth_q_frac":       t.get("birth_q_frac", 0.0),
                    "birth_shares_ahead": t.get("birth_shares_ahead", 0),
                    "birth_level_total":  t.get("birth_level_total", 0),
                }
            if t.get("terminal"):
                lt = self._live.pop(vid, {})
                lt["virtual_id"]       = vid    # needed for reward back-population
                lt["terminal_trigger"] = t["trigger"]
                lt["terminal_ts_ns"]   = t["ts_ns"]
                lt["lifetime_s"]       = (t["ts_ns"] - lt.get("birth_ts_ns", t["ts_ns"])) / 1e9
                lt["episode_return"]   = float(t.get("reward", 0.0))
                lt["q_bucket"]         = _q_frac_bucket(lt.get("birth_q_frac", 0.0))
                # repositioned=True means the order chased the market to BBO before filling.
                # fill_level_idx is the level at which the order *actually executed*:
                #   - repositioned fill → always level 0 (BBO), regardless of birth level
                #   - passive fill / cancel → birth level unchanged
                # This lets compute_sharpe_by_queue bucket the fill under the correct
                # execution level rather than the (now irrelevant) birth level.
                repo = bool(t.get("repositioned", False))
                lt["repositioned"]     = repo
                is_repo_fill = repo and t["trigger"] == TRIG_FILL
                lt["fill_level_idx"]   = 0 if is_repo_fill else lt.get("level_idx", 0)
                self.completed.append(lt)

    def compute_sharpe_by_queue(self) -> dict:
        """
        Build per-(q_bucket, birth_level, side) summary rows.

        Economic model
        ──────────────
        Orders fall into three terminal outcomes:
          1. Passive fill  (Case D — filled at birth price, no reposition)
             return = toxic_ticks           (adverse-selection cost of the fill)
          2. Repo fill     (Cases A/B — repositioned to BBO, then filled)
             return = total_ticks           (toxic + reposition contribution)
             marginal repo return = reposition_ticks (just what repositioning added)
          3. No fill       (cancelled before fill)
             return = 0.0  (no P&L)

        Sharpe columns produced
        ───────────────────────
        sharpe_filled   : Sharpe computed ONLY on filled orders (cond. on fill).
                          Splits contamination from 0-return cancels. Unbiased
                          estimate of fill quality.

        sharpe_adj      : Fill-probability-adjusted Sharpe that accounts for the
                          non-fill outcome.
                            E[R] = p_fill × μ_fill
                            Var[R] = p_fill × (σ²_fill + μ²_fill × (1−p_fill))
                          This is the correct risk-adjusted return including the
                          chance of earning nothing.

        sharpe_passive  : Sharpe across passive fills only (toxic_ticks).
                          Baseline adverse-selection quality at this queue level.

        sharpe_repo     : Sharpe across repo fills only (total_ticks).
                          Combined quality of the reposition decision + fill.

        sharpe_repo_marginal : Sharpe on reposition_ticks only.
                          Pure value-add of repositioning, stripping out the
                          toxic_ticks baseline shared with passive fills.

        Mean columns produced
        ─────────────────────
        mean_return_filled : mean total_ticks across filled orders only
        mean_passive       : mean toxic_ticks for passive fills
        mean_repo_total    : mean total_ticks for repo fills
        mean_repo_marginal : mean reposition_ticks for repo fills
        """
        from collections import defaultdict

        # Primary bucketing: by birth level (population denominator lives here)
        birth_buckets: dict = defaultdict(list)
        # Secondary bucketing: repo fills re-attributed to their execution level
        exec_buckets:  dict = defaultdict(list)

        for lt in self.completed:
            qb   = lt.get("q_bucket", "?")
            blvl = lt.get("level_idx", 0)
            flvl = lt.get("fill_level_idx", blvl)
            sd   = lt.get("side", "?")
            repo = lt.get("repositioned", False)
            trig = lt["terminal_trigger"]
            ret  = lt["episode_return"]      # total_ticks if filled, else 0
            tox  = lt.get("toxic_ticks",       np.nan)
            marg = lt.get("reposition_ticks",  np.nan)

            birth_buckets[(qb, blvl, sd)].append((ret, tox, marg, trig, repo))

            if repo and trig == TRIG_FILL and flvl != blvl:
                exec_buckets[(qb, flvl, sd)].append((ret, tox, marg))

        def _sharpe(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 2:
                return np.nan
            std = float(np.std(arr, ddof=1))
            return float(np.mean(arr)) / std if std > 1e-12 else np.nan

        def _mean(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if len(arr) else np.nan

        def _std(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan

        def _sharpe_adj(p_fill, mu_fill, sigma_fill):
            """Fill-probability-adjusted Sharpe including non-fill (return=0) outcome."""
            if np.isnan(mu_fill) or np.isnan(sigma_fill) or p_fill <= 0:
                return np.nan
            mu_adj  = p_fill * mu_fill
            var_adj = p_fill * (sigma_fill**2 + mu_fill**2 * (1.0 - p_fill))
            sigma_adj = np.sqrt(var_adj) if var_adj > 0 else np.nan
            return mu_adj / sigma_adj if sigma_adj and sigma_adj > 1e-12 else np.nan

        results = {}

        for key, records in birth_buckets.items():
            n             = len(records)
            passive_fills = [(ret, tox)  for ret, tox, marg, trig, repo in records
                             if trig == TRIG_FILL and not repo]
            repo_fills    = [(ret, marg) for ret, tox, marg, trig, repo in records
                             if trig == TRIG_FILL and repo]
            all_filled    = [ret for ret, tox, marg, trig, repo in records
                             if trig == TRIG_FILL]
            n_passive = len(passive_fills)
            n_repo    = len(repo_fills)
            n_filled  = len(all_filled)
            p_fill    = n_filled / n

            # ── Filled-only statistics ────────────────────────────────────
            mu_filled    = _mean(all_filled)
            sig_filled   = _std(all_filled)
            sh_filled    = _sharpe(all_filled)
            sh_adj       = _sharpe_adj(p_fill, mu_filled, sig_filled)

            # ── Passive-only statistics (Case D: toxic_ticks = the return) ─
            passive_tox  = [tox for _, tox in passive_fills]
            mu_passive   = _mean(passive_tox)
            sh_passive   = _sharpe(passive_tox)

            # ── Repo-only statistics ──────────────────────────────────────
            repo_total   = [ret  for ret, _ in repo_fills]   # total_ticks
            repo_marg    = [marg for _, marg in repo_fills]   # reposition_ticks only
            mu_repo_tot  = _mean(repo_total)
            mu_repo_marg = _mean(repo_marg)
            sh_repo_tot  = _sharpe(repo_total)
            sh_repo_marg = _sharpe(repo_marg)

            qb, lvl, sd = key

            def _r(v, dp=6):
                return round(float(v), dp) if v == v and v is not None else None

            results[f"birth_{key}"] = {
                "row_type":            "birth_level",
                "q_bucket":            qb,
                "birth_level_idx":     lvl + 1,
                "exec_level_idx":      None,
                "side":                sd,
                "n":                   n,
                "n_passive_fills":     n_passive,
                "n_repo_fills":        n_repo,
                "fill_rate":           round(n_filled  / n * 100, 2),
                "passive_fill_rate":   round(n_passive / n * 100, 2),
                "repo_fill_rate":      round(n_repo    / n * 100, 2),
                # ── Filled-only return (no contamination from 0-return cancels) ──
                "mean_return_filled":  _r(mu_filled),
                "std_return_filled":   _r(sig_filled),
                "sharpe_filled":       _r(sh_filled,   4),
                # ── Fill-probability-adjusted Sharpe (economically correct) ──────
                "sharpe_adj":          _r(sh_adj,      4),
                # ── Passive fills: baseline adverse-selection quality ────────────
                "mean_passive":        _r(mu_passive),
                "sharpe_passive":      _r(sh_passive,  4),
                # ── Repo fills: total quality + marginal repo contribution ────────
                "mean_repo_total":     _r(mu_repo_tot),
                "sharpe_repo":         _r(sh_repo_tot, 4),
                "mean_repo_marginal":  _r(mu_repo_marg),
                "sharpe_repo_marginal":_r(sh_repo_marg,4),
                # ── Legacy (blended, includes 0-returns; kept for compat) ─────────
                "mean_return":         _r(float(np.mean([r for r, *_ in records]))),
                "std_return":          _r(float(np.std([r for r, *_ in records], ddof=1)) if n > 1 else np.nan),
                "sharpe":              _r(_sharpe([r for r, *_ in records]), 4),
            }

        # Execution-level rows: repo fills counted at the level they actually traded
        for key, rets in exec_buckets.items():
            totals = [ret  for ret, tox, marg in rets]
            margs  = [marg for ret, tox, marg in rets]
            n      = len(totals)
            mu     = _mean(totals)
            sig    = _std(totals)
            sh     = _sharpe(totals)
            sh_m   = _sharpe(margs)
            qb, lvl, sd = key

            results[f"exec_{key}"] = {
                "row_type":             "exec_level",
                "q_bucket":             qb,
                "birth_level_idx":      None,
                "exec_level_idx":       lvl + 1,
                "side":                 sd,
                "n":                    n,
                "n_passive_fills":      0,
                "n_repo_fills":         n,
                "fill_rate":            100.0,
                "passive_fill_rate":    0.0,
                "repo_fill_rate":       100.0,
                # All exec-level orders are filled by construction
                "mean_return_filled":   _r(mu),
                "std_return_filled":    _r(sig),
                "sharpe_filled":        _r(sh, 4),
                "sharpe_adj":           _r(sh, 4),   # p_fill=1.0, adj = filled
                "mean_passive":         None,
                "sharpe_passive":       None,
                "mean_repo_total":      _r(mu),
                "sharpe_repo":          _r(sh, 4),
                "mean_repo_marginal":   _r(_mean(margs)),
                "sharpe_repo_marginal": _r(sh_m, 4),
                # Legacy
                "mean_return":          _r(mu),
                "std_return":           _r(sig),
                "sharpe":               _r(sh, 4),
            }

        return results

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "virtual_id":         lt.get("virtual_id", ""),
            "date":               lt.get("birth_date",""),
            "side":               lt.get("side",""),
            "birth_level_idx":    lt.get("level_idx",0) + 1,       # 1-based birth level
            "fill_level_idx":     lt.get("fill_level_idx",
                                         lt.get("level_idx",0)) + 1, # 1-based execution level
            "repositioned":       lt.get("repositioned", False),
            "q_bucket":           lt.get("q_bucket",""),
            "birth_q_frac":       lt.get("birth_q_frac",0.0),
            "birth_shares_ahead": lt.get("birth_shares_ahead",0),
            "birth_level_total":  lt.get("birth_level_total",0),
            "birth_mid":          lt.get("birth_mid",np.nan),
            "terminal_trigger":   lt.get("terminal_trigger",""),
            "lifetime_s":         lt.get("lifetime_s",0.0),
            "episode_return":     lt.get("episode_return",0.0),
        } for lt in self.completed])


# ─────────────────────────────────────────────────────────────────────────────

def _stamp_meta(t: dict, meta: dict, session_id: int) -> None:
    """In-place: attach placement metadata and session_id to an event dict."""
    t["session_id"]         = session_id
    t["level_idx"]          = meta.get("level_idx",          0)
    t["birth_q_frac"]       = meta.get("birth_q_frac",       0.0)
    t["birth_shares_ahead"] = meta.get("birth_shares_ahead",  0)
    t["birth_level_total"]  = meta.get("birth_level_total",   0)


# ─────────────────────────────────────────────────────────────────────────────
# Mid-price ring buffer  (streaming replacement for full-day arrays)
# ─────────────────────────────────────────────────────────────────────────────
# Old: two np.nan arrays of size N (22M rows = ~176MB each) + binary search.
# New: rolling 2000ms deque of (ts_ns, mid) pairs — O(messages in 2s window)
# regardless of day length.  Covers both the +100ms and -100ms lookups.

from collections import deque as _deque
from bisect    import bisect_left as _bisect_left, bisect_right as _bisect_right

class _MidRingBuffer:
    # FIX-21d: 2000ms window (was 200ms).
    # mid_100ms_ago needs the mid at ts_cancel - 100ms.  lookup_backward searches
    # within [ts - BUFFER_NS, ts - 100ms].  With BUFFER_NS=200ms this window was
    # only 100ms wide, and in quiet markets (no book updates for >200ms) the buffer
    # could be empty, causing 38% NaN rates.  2000ms gives 1900ms of history to
    # search, covering all but the most illiquid periods with negligible extra RAM
    # (each entry is ~24 bytes; 2s of ITCH data ≈ a few thousand entries at most).
    BUFFER_NS = 2_000_000_000   # 2 000 ms

    def __init__(self) -> None:
        # Deques for O(1) left-trim; parallel ts/mid lists for bisect lookups.
        from collections import deque as _deque
        self._ts_dq:  "deque[int]"   = _deque()
        self._mid_dq: "deque[float]" = _deque()
        self._ts:  List[int]   = []   # snapshot list for bisect (rebuilt from deque)
        self._mid: List[float] = []
        self._ts_dirty: bool   = False
        self._bid: Dict[float, int] = {}
        self._ask: Dict[float, int] = {}
        # Cached best bid/ask — updated incrementally to avoid max()/min() every call
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        self._bid_dirty: bool = False  # best_bid needs recompute
        self._ask_dirty: bool = False

    def _sync_lists(self) -> None:
        """Rebuild the bisect-able lists from the deques when dirty."""
        self._ts  = list(self._ts_dq)
        self._mid = list(self._mid_dq)
        self._ts_dirty = False

    def update(self, ts_ns: int, msg_type: str, side: str,
               price: float, shares: int) -> None:
        # Only process genuine bid/ask messages; ignore non-displayed etc.
        if side not in ("B", "A"):
            return
        if side == "B":
            bk = self._bid
            if msg_type == "A":
                prev = bk.get(price, 0)
                bk[price] = prev + shares
                if self._best_bid is None or price > self._best_bid:
                    self._best_bid = price
            elif msg_type in ("C", "M", "T", "F"):
                rem = bk.get(price, 0) - shares
                if rem <= 0:
                    bk.pop(price, None)
                    if price == self._best_bid:
                        self._bid_dirty = True
                        self._best_bid = max(bk) if bk else None
                        self._bid_dirty = False
                else:
                    bk[price] = rem
        else:
            bk = self._ask
            if msg_type == "A":
                prev = bk.get(price, 0)
                bk[price] = prev + shares
                if self._best_ask is None or price < self._best_ask:
                    self._best_ask = price
            elif msg_type in ("C", "M", "T", "F"):
                rem = bk.get(price, 0) - shares
                if rem <= 0:
                    bk.pop(price, None)
                    if price == self._best_ask:
                        self._ask_dirty = True
                        self._best_ask = min(bk) if bk else None
                        self._ask_dirty = False
                else:
                    bk[price] = rem

        bb  = self._best_bid
        ba  = self._best_ask
        mid = (bb + ba) * 0.5 if (bb is not None and ba is not None) else np.nan
        self._ts_dq.append(ts_ns)
        self._mid_dq.append(mid)
        self._ts_dirty = True

        # Expire entries older than the window using O(1) popleft
        cutoff = ts_ns - self.BUFFER_NS
        while self._ts_dq and self._ts_dq[0] < cutoff:
            self._ts_dq.popleft()
            self._mid_dq.popleft()

    def lookup_forward(self, ts_ns: int, offset_ns: int) -> float:
        """First mid with timestamp >= ts_ns + offset_ns, else NaN."""
        if self._ts_dirty:
            self._sync_lists()
        target = ts_ns + offset_ns
        idx = _bisect_left(self._ts, target)
        for i in range(idx, len(self._ts)):
            m = self._mid[i]
            if m == m:   # fast NaN check: NaN != NaN
                return m
        return np.nan

    def lookup_backward(self, ts_ns: int, offset_ns: int) -> float:
        """Last mid with timestamp <= ts_ns - offset_ns, else NaN."""
        if self._ts_dirty:
            self._sync_lists()
        target = ts_ns - offset_ns
        idx = _bisect_right(self._ts, target) - 1
        for i in range(idx, -1, -1):
            m = self._mid[i]
            if m == m:
                return m
        return np.nan

    def reset(self) -> None:
        self._ts_dq.clear(); self._mid_dq.clear()
        self._ts = []; self._mid = []
        self._bid.clear(); self._ask.clear()
        self._best_bid = None; self._best_ask = None
        self._ts_dirty = False




# Stage 3 + 4: Contribution calculation + Summary statistics
# ─────────────────────────────────────────────────────────────────────────────



# stage4_summary removed — cross-day summary is now handled by
# summarise_pipeline.py which reads the per-day CSV outputs directly.
# run_fqi_analysis is no longer called from the main pipeline.


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime / queue-position reporting  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _save_lifetime_summary(
    lifetime: OrderLifetimeTracker,
    out: Path,
    save_prefix: str,
    chunk_label: str,
) -> None:
    """
    Save per-chunk lifetime and Sharpe outputs.
    chunk_label is embedded in the filename so each day's file is distinct
    and summarise_pipeline.py can discover them all by glob.
    """
    n     = len(lifetime.completed)
    fills = sum(1 for lt in lifetime.completed if lt.get("terminal_trigger") == TRIG_FILL)
    logger.info("Lifetime: %d orders completed (%d fills = %.1f%%)",
                n, fills, fills / max(n, 1) * 100)

    # Sharpe by queue bucket
    sharpe_rows = list(lifetime.compute_sharpe_by_queue().values())
    if sharpe_rows:
        sharpe_df   = pd.DataFrame(sharpe_rows).sort_values(
            ["row_type", "birth_level_idx", "exec_level_idx", "q_bucket", "side"],
            na_position="last",
        )
        sharpe_path = out / f"{save_prefix}_{chunk_label}_sharpe_by_queue.csv"
        sharpe_df.to_csv(sharpe_path, index=False)
        del sharpe_df
        logger.info("Sharpe by queue saved → %s", sharpe_path)

    lt_df = lifetime.to_dataframe()
    if not lt_df.empty:
        lt_path = out / f"{save_prefix}_{chunk_label}_lifetime_orders.parquet"
        lt_df.to_parquet(lt_path, index=False)
        del lt_df
        logger.info("Lifetime orders saved → %s", lt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────



def process_chunk(
    parquet_path:   Path,
    day_min:        int,
    day_max:        int,
    chunk_label:    str,
    out:            Path,
    save_prefix:    str,
    tick_size:      float,
    warmup_minutes: int,
    seed:           int,
    batch_size:     int = 100_000,
) -> None:
    """
    Stream one calendar-day window row-by-row using PyArrow row-group batches.

    Memory model
    ------------
    PyArrow yields one RecordBatch (~batch_size rows) at a time.  Each batch
    is converted to pandas, filtered to this day's date range, iterated
    row-by-row, then explicitly deleted before the next batch is loaded.
    At no point is more than one batch in RAM simultaneously.

    Peak RAM:
        1 batch  ~  batch_size * ~200 bytes  =  ~20 MB at 100k rows
        tracker state  ~  n_active_orders * small dataclass  (tiny)
        mid ring buffer  ~  messages in 2s window  (small — ~24 bytes/entry)

    Correctness
    -----------
    - std_to_shd and its inverse are maintained incrementally; the inverse
      is never rebuilt from scratch inside the hot loop.
    - Session boundaries are detected per-row (30-min gap) and trigger a
      full book + state reset before the new session begins.
    - Mid back-fill only rescans unresolved events (those still holding NaN)
      so cost stays O(unresolved) not O(all_events).
    - warmup is tracked per-session; placement is gated by session_start_ns.

    Returns (fill_rows, cancel_rows).
    """
    import random
    import time as _time
    import datetime as _datetime
    rng = random.Random(seed)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _ts_to_hms(ns: int) -> str:
        """Format a nanosecond timestamp as HH:MM:SS UTC."""
        try:
            return _datetime.datetime.utcfromtimestamp(ns / 1e9).strftime("%H:%M:%S")
        except Exception:
            return "??:??:??"

    def _fmt_rate(n: int, elapsed: float) -> str:
        """Format a throughput number as k/s or M/s."""
        if elapsed < 1e-6:
            return "—"
        r = n / elapsed
        if r >= 1_000_000:
            return f"{r/1_000_000:.2f}M/s"
        return f"{r/1_000:.1f}k/s"

    # ── Persistent state (lives across all batches within this chunk) ─────
    std_tracker = VirtualOrderTracker(mode=MODE_STANDARD, queue_change_threshold=0.10)
    shd_tracker = VirtualOrderTracker(mode=MODE_SHADOW,   queue_change_threshold=0.10)
    lifetime    = OrderLifetimeTracker()
    mid_buf     = _MidRingBuffer()

    warmup_ns      = warmup_minutes * 60 * 10**9
    SESSION_GAP_NS = 30 * 60 * 10**9
    MAX_CONCURRENT = 100
    DELAY_MAX_NS   = 1 * 10**9   # 0-1s jitter; avg 500ms between placements per slot

    # Placement state
    next_place_ns:   Dict[str, Optional[int]] = {"B": None, "A": None}
    active_std_vids: Dict[str, set]           = {"B": set(), "A": set()}
    std_to_shd:      Dict[str, str]           = {}   # std_vid  → shd_vid
    shd_to_std:      Dict[str, str]           = {}   # shd_vid  → std_vid  (inverse, kept in sync)
    # Orphan shadows: shadows that survive after their paired std was CANCELLED.
    # Key = shd_vid, Value = std_vid they belong to.
    # They keep receiving on_message() feeds (shd_tracker holds them) and can
    # still fill naturally until session close. Cleared at session boundary.
    orphan_shd:      Dict[str, str]           = {}   # shd_vid  → std_vid
    # Same-message remap: when std cancels and shadow fills on the SAME message,
    # _retire_std sees the shadow already gone from _virtual_orders so it cannot
    # register an orphan. This thin map carries the shd_vid→std_vid mapping just
    # long enough for the new_shd loop (same iteration) to remap the fill event.
    # Entries are consumed (popped) on first use and never accumulate.
    pending_shd_remap: Dict[str, str]         = {}   # shd_vid  → std_vid (transient)
    vo_meta:         Dict[str, dict]          = {}   # std_vid  → placement metadata

    # Session state
    session_start_ns: int = 0   # ts_ns of first message in current session
    current_sid:      int = 0   # incremented at each 30-min gap
    prev_ts_ns:       int = 0   # last message timestamp

    # Accumulators
    std_events:  List[dict] = []
    shd_events:  List[dict] = []
    # Indices into std_events / shd_events of events whose mid fields are
    # still NaN — we only rescan these on each back-fill pass.
    _unresolved_std: List[int] = []
    _unresolved_shd: List[int] = []

    # Counters
    n_placed    = 0
    n_std_fills = 0
    n_std_cxls  = 0
    n_shd_fills = 0
    n_warmup_skips = 0

    chunk_wall_t0 = _time.monotonic()

    # Pre-generated order-size pool: numpy batch generation is ~20× faster than
    # repeated rng.gauss() calls.  Refilled automatically when exhausted.
    _SIZE_POOL_N  = 2_000
    _size_pool:   list = []
    _size_pool_i: int  = 0

    def _next_size() -> int:
        nonlocal _size_pool, _size_pool_i
        if _size_pool_i >= len(_size_pool):
            raw = np.random.normal(200, 150, _SIZE_POOL_N)
            _size_pool   = [max(1, int(x)) for x in raw]
            _size_pool_i = 0
        v = _size_pool[_size_pool_i]
        _size_pool_i += 1
        return v

    # ── Inner helpers ─────────────────────────────────────────────────────

    def _schedule(side: str, after_ns: int) -> None:
        next_place_ns[side] = after_ns + rng.randint(0, DELAY_MAX_NS)

    def _place_pair(ts_ns: int, side: str) -> Optional[str]:
        nonlocal n_placed
        result = _sample_queue_placement(std_tracker, side, rng, max_levels=10)
        if result is None:
            return None
        price, total_sa, sa_in_level, level_idx, level_total = result
        dyn_size = _next_size()

        # Compute shares_strictly_ahead once — same for both trackers at placement time
        _book_for_sa = std_tracker._bid_book if side == "B" else std_tracker._ask_book
        _sa_at_place = float(_book_for_sa.shares_strictly_ahead(price))
        std_vid = std_tracker.place_order(side=side, price=price, size=dyn_size, ts_ns=ts_ns,
                                          shares_ahead=_sa_at_place)
        shd_vid = shd_tracker.place_order(side=side, price=price, size=dyn_size, ts_ns=ts_ns,
                                          shares_ahead=_sa_at_place)

        # Maintain both directions of the vid map incrementally
        std_to_shd[std_vid] = shd_vid
        shd_to_std[shd_vid] = std_vid
        active_std_vids[side].add(std_vid)
        n_placed += 1
        # Placement timing is now managed entirely by the hot-loop two-mode gate:
        #   - Recovery mode (pool < MAX_CONCURRENT): places on every row, no timer.
        #   - Steady-state (pool == MAX_CONCURRENT): next fill/cancel will remove a slot,
        #     dropping pool to MAX_CONCURRENT-1 and re-enabling the hot-loop gate.
        # No _schedule() call needed here.

        q_frac = sa_in_level / max(level_total, 1)
        meta   = {
            "level_idx":          level_idx,
            "birth_q_frac":       q_frac,
            "birth_shares_ahead": total_sa,
            "birth_level_total":  level_total,
            "order_size":         dyn_size,
            "session_id":         current_sid,
            "side":               side,        # stored for slot-recovery in _retire_std
        }
        vo_meta[std_vid] = meta
        # meta is already stored in vo_meta[std_vid].
        # The pipeline always accesses placement metadata via vo_meta, not via
        # VirtualOrder attributes — the __dict__.update() was redundant.
        return std_vid

    def _retire_std(vid: str, ts_ns: int, std_trigger: str) -> None:
        """Clean up a terminal standard order.

        Shadow lifecycle depends on WHY the standard terminated:

          std FILLED   → cancel shadow immediately (both filled; comparison done)
          std CANCELLED → let shadow SURVIVE as an orphan so it can still fill
                          naturally, answering "would staying have been better?"
                          Orphan shadows keep receiving on_message() and are only
                          cleaned up at session close (_close_session).
        """
        side = None
        for s in ("B", "A"):
            if vid in active_std_vids[s]:
                active_std_vids[s].discard(vid)
                side = s
                break

        if side is None:
            # vid not found in active_std_vids — duplicate terminal event.
            # Recover the side from vo_meta so the slot is still rescheduled.
            side = vo_meta.get(vid, {}).get("side")
            if side:
                logger.debug("_retire_std: vid %s not in active_std_vids; "
                             "recovered side=%s from vo_meta (duplicate terminal?)", vid[:8], side)

        if side:
            # In recovery mode (pool < MAX_CONCURRENT), the hot loop places immediately
            # on every message — no timer needed.  In steady-state (pool == MAX_CONCURRENT),
            # the hot loop's gate (n_active >= MAX_CONCURRENT) blocks placement;
            # after this slot is removed from active_std_vids the gate will allow one
            # placement on the very next message without any explicit timer.
            # So we no longer need to set next_place_ns here; clearing it lets the
            # hot-loop recovery path fire on the next incoming row.
            next_place_ns[side] = None   # signal: place as soon as possible
        else:
            logger.warning("_retire_std: vid %s — side unknown, slot may be lost", vid[:8])

        shd_vid = std_to_shd.pop(vid, None)
        if shd_vid:
            if std_trigger == TRIG_FILL:
                # Std filled → cancel shadow immediately; comparison is fill vs fill.
                #
                # IMPORTANT: do NOT pop shd_to_std here.
                # Both std and shd emit FILL on the same message (both grace periods
                # expire together). new_std is processed first, so _retire_std runs
                # before the new_shd loop.  If we pop shd_to_std now, the new_shd
                # loop cannot find the std_vid to remap the shadow fill event, causing
                # the "could not resolve std_vid" warning and a corrupted virtual_id.
                #
                # shd_to_std is cleaned up by the new_shd loop when it processes the
                # terminal shadow event (see "if t.get('terminal')" block below).
                if shd_vid in shd_tracker._virtual_orders:
                    sc = shd_tracker.cancel_order(shd_vid, ts_ns)
                    if sc:
                        _stamp_meta(sc, vo_meta.get(vid, {}), current_sid)
                        sc["virtual_id"] = vid
                        idx = len(shd_events)
                        shd_events.append(sc)
                        if np.isnan(float(sc.get("mid_post_100ms", np.nan))):
                            _unresolved_shd.append(idx)

            else:
                # CANCEL path: pop shd_to_std now — orphan_shd takes over the mapping.
                shd_to_std.pop(shd_vid, None)
                # Std cancelled → let shadow survive until session close (if still alive).
                if shd_vid in shd_tracker._virtual_orders:
                    # Shadow is still live — register as orphan.
                    # Register as orphan so _close_session and the shd event loop
                    # can still identify which std_vid this shadow belongs to.
                    orphan_shd[shd_vid] = vid
                    logger.debug(
                        "_retire_std: std %s cancelled — shadow %s survives as orphan",
                        vid[:8], shd_vid[:8],
                    )
                    # Snapshot meta now (vo_meta will be popped below) onto a
                    # shd_vid-keyed entry so _close_session can stamp events
                    # correctly even after vo_meta[vid] is gone.
                    vo_meta[shd_vid] = dict(vo_meta.get(vid, {}))   # shallow copy
                else:
                    # Shadow already filled on this same message (same-message race):
                    # shd_tracker processed the fill before _retire_std ran and
                    # removed the shadow from _virtual_orders.  The fill event is
                    # already in new_shd with virtual_id=shd_vid (its own id).
                    # Record a transient remap so the new_shd loop can correct it.
                    pending_shd_remap[shd_vid] = vid
                    # Also snapshot meta under shd_vid key for the new_shd stamping.
                    vo_meta[shd_vid] = dict(vo_meta.get(vid, {}))

        # Pop std meta only after shadow logic above has read it
        vo_meta.pop(vid, None)

    def _close_session(ts_ns: int) -> None:
        """Cancel all open orders and reset books at a session boundary."""
        n_open = sum(len(v) for v in active_std_vids.values())
        logger.info(
            "  [sid=%d] Session close at %s — cancelling %d open orders",
            current_sid, _ts_to_hms(ts_ns), n_open,
        )
        for side in ("B", "A"):
            for vid in list(active_std_vids[side]):
                ct = std_tracker.cancel_order(vid, ts_ns)
                if ct:
                    _stamp_meta(ct, vo_meta.get(vid, {}), current_sid)
                    idx = len(std_events)
                    std_events.append(ct)
                    lifetime.record([ct])
                    if np.isnan(float(ct.get("mid_post_100ms", np.nan))):
                        _unresolved_std.append(idx)
                shd_vid = std_to_shd.pop(vid, None)
                if shd_vid:
                    shd_to_std.pop(shd_vid, None)
                    if shd_vid in shd_tracker._virtual_orders:
                        sc = shd_tracker.cancel_order(shd_vid, ts_ns)
                        if sc:
                            _stamp_meta(sc, vo_meta.get(vid, {}), current_sid)
                            sc["virtual_id"] = vid
                            idx = len(shd_events)
                            shd_events.append(sc)
                            if np.isnan(float(sc.get("mid_post_100ms", np.nan))):
                                _unresolved_shd.append(idx)
                vo_meta.pop(vid, None)
            active_std_vids[side].clear()

        # ── Flush orphan shadows (survived after their std was cancelled) ──
        # These are shadows that have been alive since their paired std was
        # QPR-cancelled.  At session close they must be terminated so they
        # appear in shd_events (as CANCEL with terminal=True) for
        # calculate_cancel_value to process them.
        n_orphans = len(orphan_shd)
        if n_orphans:
            logger.info("  [sid=%d] Closing %d orphan shadows at session boundary",
                        current_sid, n_orphans)
        for shd_vid, std_vid in list(orphan_shd.items()):
            if shd_vid in shd_tracker._virtual_orders:
                sc = shd_tracker.cancel_order(shd_vid, ts_ns)
                if sc:
                    _stamp_meta(sc, vo_meta.get(shd_vid, {}), current_sid)
                    sc["virtual_id"] = std_vid   # re-attach to std so cancel_value finds it
                    idx = len(shd_events)
                    shd_events.append(sc)
                    if np.isnan(float(sc.get("mid_post_100ms", np.nan))):
                        _unresolved_shd.append(idx)
            vo_meta.pop(shd_vid, None)   # clean up the snapshot stored by _retire_std
        orphan_shd.clear()

        std_tracker.reset_book()
        shd_tracker.reset_book()
        mid_buf.reset()
        next_place_ns["B"] = None
        next_place_ns["A"] = None
        logger.info(
            "  [sid=%d] Books reset. std_events=%d shd_events=%d",
            current_sid, len(std_events), len(shd_events),
        )

    def _fill_mid_unresolved(ts_ns: int) -> None:
        """
        Rescan unresolved events using the ring buffer.

        mid_100ms_ago is now attempted immediately on emission, so only
        genuine forward-lookup (FILL mid_post_100ms) and rare backward
        misses should remain here.

        TTL rule: an event older than BUFFER_NS from current ts_ns will
        never resolve (the buffer has evicted its window). Accept the NaN
        and remove it to prevent unbounded list growth.
        """
        ttl_cutoff = ts_ns - mid_buf.BUFFER_NS   # events older than this: give up

        still_std: List[int] = []
        for idx in _unresolved_std:
            t = std_events[idx]
            evt_ts = int(t["ts_ns"])

            # TTL: event too old for buffer to ever resolve — leave as NaN, drop
            if evt_ts < ttl_cutoff:
                continue

            still_unresolved = False

            # mid_post_100ms — FILL events only (forward = future data)
            if t.get("trigger") == TRIG_FILL and np.isnan(float(t.get("mid_post_100ms", np.nan))):
                val = mid_buf.lookup_forward(evt_ts, 100_000_000)
                if not np.isnan(val):
                    t["mid_post_100ms"] = val
                else:
                    still_unresolved = True

            # mid_100ms_ago — retry if emission-time attempt failed
            if np.isnan(float(t.get("mid_100ms_ago", np.nan))):
                val = mid_buf.lookup_backward(evt_ts, 100_000_000)
                if not np.isnan(val):
                    t["mid_100ms_ago"] = val
                else:
                    still_unresolved = True

            if still_unresolved:
                still_std.append(idx)
        _unresolved_std[:] = still_std

        still_shd: List[int] = []
        for idx in _unresolved_shd:
            t = shd_events[idx]
            evt_ts = int(t["ts_ns"])

            if evt_ts < ttl_cutoff:
                continue

            still_unresolved = False

            if t.get("trigger") == TRIG_FILL and np.isnan(float(t.get("mid_post_100ms", np.nan))):
                val = mid_buf.lookup_forward(evt_ts, 100_000_000)
                if not np.isnan(val):
                    t["mid_post_100ms"] = val
                else:
                    still_unresolved = True

            if np.isnan(float(t.get("mid_100ms_ago", np.nan))):
                val = mid_buf.lookup_backward(evt_ts, 100_000_000)
                if not np.isnan(val):
                    t["mid_100ms_ago"] = val
                else:
                    still_unresolved = True

            if still_unresolved:
                still_shd.append(idx)
        _unresolved_shd[:] = still_shd

    # ── Open parquet and stream ───────────────────────────────────────────
    import pyarrow.parquet as pq

    pf         = pq.ParquetFile(str(parquet_path))
    day_min_ns = int(day_min) * 86_400 * 10**9
    day_max_ns = (int(day_max) + 1) * 86_400 * 10**9   # exclusive upper bound

    total_rows     = 0
    batch_idx      = 0

    logger.info(
        "╔══ Chunk %s ══╗  days %d–%d  batch_size=%d",
        chunk_label, day_min, day_max, batch_size,
    )

    for arrow_batch in pf.iter_batches(batch_size=batch_size):
        batch_pd = arrow_batch.to_pandas()
        del arrow_batch

        # Filter to this chunk's date range
        ts_col   = batch_pd["timestamp"].astype("int64")
        mask     = (ts_col >= day_min_ns) & (ts_col < day_max_ns)
        del ts_col
        batch_pd = batch_pd[mask]   # no reset_index — itertuples(index=False) doesn't need it
        del mask
        if batch_pd.empty:
            del batch_pd
            continue

        batch_idx    += 1
        n_batch       = len(batch_pd)
        batch_wall_t0 = _time.monotonic()
        ts_ns_arr    = batch_pd["timestamp"].astype("int64").to_numpy()
        col_names    = list(batch_pd.columns)
        # Build index-lookup dict once per batch — avoids dict(zip()) every row
        _ci = {c: i for i, c in enumerate(col_names)}
        _i_ts    = _ci["timestamp"]
        _i_mtype = _ci["message_type"]
        _i_side  = _ci["side"]
        _i_price = _ci["price_float"]
        _i_share = _ci["shares"]
        _i_oid   = _ci["order_id"]
        # Set of message types that affect the book (skip mid_buf for others)
        _BOOK_MSG = frozenset(("A", "C", "M", "T", "F"))

        n_placed_batch   = 0

        for i, raw_tuple in enumerate(batch_pd.itertuples(index=False, name=None)):
            ts_ns     = ts_ns_arr[i]
            msg_type  = raw_tuple[_i_mtype]
            side      = raw_tuple[_i_side]
            price     = float(raw_tuple[_i_price])
            shares    = int(raw_tuple[_i_share])
            total_rows += 1

            # ── Mid buffer (must happen before session/placement logic) ───
            # Only feed messages that actually change the book
            if msg_type in _BOOK_MSG:
                mid_buf.update(ts_ns, msg_type, side, price, shares)

            # ── Session boundary detection ────────────────────────────────
            if prev_ts_ns > 0 and (ts_ns - prev_ts_ns) > SESSION_GAP_NS:
                gap_min = (ts_ns - prev_ts_ns) / 6e10
                logger.info(
                    "  *** Session boundary detected at %s (gap=%.1f min) "
                    "sid %d → %d ***",
                    _ts_to_hms(ts_ns), gap_min, current_sid, current_sid + 1,
                )
                _close_session(prev_ts_ns)
                current_sid      += 1
                session_start_ns  = ts_ns
                logger.info(
                    "  [sid=%d] New session starts at %s",
                    current_sid, _ts_to_hms(ts_ns),
                )
            elif session_start_ns == 0:
                # Very first message of this chunk
                session_start_ns = ts_ns
                logger.info(
                    "  [sid=%d] Session starts at %s",
                    current_sid, _ts_to_hms(ts_ns),
                )

            prev_ts_ns = ts_ns

            # ── Warmup gate ───────────────────────────────────────────────
            past_warmup = ts_ns >= session_start_ns + warmup_ns
            if not past_warmup:
                n_warmup_skips += 1
                # Feed trackers even during warmup so the book is populated
                _oid_str = str(raw_tuple[_i_oid])
                std_tracker.on_message({}, ts_ns, msg_type, side, price, shares, _oid_str)
                shd_tracker.on_message({}, ts_ns, msg_type, side, price, shares, _oid_str)
                continue

            # ── Order placement ───────────────────────────────────────────
            # NOTE: loop var is _pl_side NOT side — avoids clobbering the
            # message's side field which on_message needs immediately after.
            #
            # Two-mode placement gate:
            #
            #   RECOVERY mode  (pool < MAX_CONCURRENT):
            #     Bypass the timer entirely and place one order per message per side.
            #     The 1-per-row-per-side rate naturally limits burst to ~rows/s placements,
            #     which is far faster than fills can drain the pool. This ensures the
            #     pool recovers to capacity after a burst drain.
            #
            #     Without this, frequent fills keep resetting the timer before it fires —
            #     if fills arrive faster than DELAY_MAX_NS//10 (100ms), every _retire_std
            #     overwrites the timer and it never expires. Pool stays depleted indefinitely.
            #
            #   STEADY-STATE mode (pool == MAX_CONCURRENT):
            #     Timer-gated: each filled/cancelled slot schedules its own replacement
            #     with a 0–100ms delay (set by _retire_std), spreading placements over time
            #     and preventing correlated bursts of simultaneous order expiry.
            for _pl_side in ("B", "A"):
                n_active = len(active_std_vids[_pl_side])
                if n_active >= MAX_CONCURRENT:
                    continue
                # Recovery mode: pool is below capacity — place immediately, no timer.
                # _place_pair itself calls _schedule to set the next steady-state timer,
                # but we don't add an extra _schedule call here (avoids double-schedule
                # overwriting the timer with a longer delay immediately after placement).
                vid = _place_pair(ts_ns, _pl_side)
                if vid is not None:
                    n_placed_batch += 1

            # ── Tracker processing ────────────────────────────────────────
            # Pass pre-extracted scalars — skip all dict lookups inside on_message
            _oid_str = str(raw_tuple[_i_oid])
            new_std = std_tracker.on_message({}, ts_ns, msg_type, side, price, shares, _oid_str)
            new_shd = shd_tracker.on_message({}, ts_ns, msg_type, side, price, shares, _oid_str)

            # Stamp and register standard events
            for t in new_std:
                vid  = t["virtual_id"]
                meta = vo_meta.get(vid, {})
                _stamp_meta(t, meta, current_sid)
                vo_live = std_tracker._virtual_orders.get(vid)
                t["order_size"] = int(vo_live.size) if vo_live else meta.get("order_size", 1)
                idx = len(std_events)
                std_events.append(t)
                lifetime.record([t])
                # Try to resolve mid_100ms_ago immediately (backward = past data in buffer).
                # Only add to unresolved if backward fails or this is a FILL needing forward.
                _evt_ts = int(t["ts_ns"])
                _needs_resolution = False
                _mago = t.get("mid_100ms_ago", np.nan)
                if _mago != _mago:   # fast NaN check (NaN != NaN)
                    _bval = mid_buf.lookup_backward(_evt_ts, 100_000_000)
                    if _bval == _bval:
                        t["mid_100ms_ago"] = _bval
                    else:
                        _needs_resolution = True
                if t.get("trigger") == TRIG_FILL and t.get("mid_post_100ms", np.nan) != t.get("mid_post_100ms", np.nan):
                    _needs_resolution = True   # forward lookup = future data, always defer
                if _needs_resolution:
                    _unresolved_std.append(idx)
                if t.get("terminal"):
                    if t["trigger"] == TRIG_FILL:
                        n_std_fills += 1
                    else:
                        n_std_cxls  += 1
                    _retire_std(vid, ts_ns, t["trigger"])

            # Stamp and register shadow events
            for t in new_shd:
                shd_vid = t["virtual_id"]
                # Resolve shd_vid → std_vid via three maps, in priority order:
                #
                #  1. shd_to_std (pop on terminal) — active pair, std still alive OR
                #     std just filled on this same message.  _retire_std deliberately
                #     leaves shd_to_std intact for TRIG_FILL so we can pop it here
                #     AFTER remapping — avoiding the "std processed first" race where
                #     popping early left the shadow unresolvable.
                #
                #  2. orphan_shd — std was cancelled; shadow survived and is still live.
                #     Entry popped when shadow terminates (fill or session close).
                #
                #  3. pending_shd_remap — same-message race for cancel+fill: std cancel
                #     fires, shadow happened to also fill in the same on_message() call,
                #     shadow was already gone from _virtual_orders so orphan path skipped.
                #     Entry consumed (popped) on first use; transient, never accumulates.
                is_terminal = t.get("terminal", False)

                if shd_vid in shd_to_std:
                    # Active-pair path (covers simultaneous fill — std processed first
                    # but intentionally left shd_to_std intact until now).
                    std_vid = shd_to_std[shd_vid]
                    if is_terminal:
                        shd_to_std.pop(shd_vid, None)   # deferred cleanup
                elif shd_vid in orphan_shd:
                    std_vid = orphan_shd[shd_vid]
                    if is_terminal:
                        orphan_shd.pop(shd_vid, None)
                        vo_meta.pop(shd_vid, None)
                elif shd_vid in pending_shd_remap:
                    std_vid = pending_shd_remap.pop(shd_vid)   # consumed on use
                else:
                    std_vid = shd_vid   # should never reach; warning below
                    if is_terminal:
                        logger.warning(
                            "new_shd: could not resolve std_vid for shd %s "
                            "(trigger=%s) — vid remapping failed; "
                            "fill/cancel_value row will be indexed under shadow id",
                            shd_vid[:8], t.get("trigger"),
                        )

                meta = vo_meta.get(std_vid, vo_meta.get(shd_vid, {}))
                _stamp_meta(t, meta, current_sid)
                t["virtual_id"] = std_vid
                vo_live = shd_tracker._virtual_orders.get(shd_vid)
                t["order_size"] = int(vo_live.size) if vo_live else meta.get("order_size", 1)
                idx = len(shd_events)
                shd_events.append(t)
                _evt_ts = int(t["ts_ns"])
                _needs_resolution = False
                if np.isnan(float(t.get("mid_100ms_ago", np.nan))):
                    _bval = mid_buf.lookup_backward(_evt_ts, 100_000_000)
                    if not np.isnan(_bval):
                        t["mid_100ms_ago"] = _bval
                    else:
                        _needs_resolution = True
                if t.get("trigger") == TRIG_FILL and np.isnan(float(t.get("mid_post_100ms", np.nan))):
                    _needs_resolution = True
                if _needs_resolution:
                    _unresolved_shd.append(idx)
                if t.get("terminal") and t["trigger"] == TRIG_FILL:
                    n_shd_fills += 1

        # ── End of batch ──────────────────────────────────────────────────
        # Back-fill mid fields for events emitted in this batch.
        # Only rescans events that still have NaN — cost is O(unresolved).
        _fill_mid_unresolved(ts_ns)

        del batch_pd   # ← free pandas DataFrame before next batch loads
        batch_wall_t1  = _time.monotonic()
        del ts_ns_arr, col_names  # row-loop helpers no longer needed

        # Per-batch progress report — every batch for timing visibility
        batch_elapsed  = batch_wall_t1 - batch_wall_t0
        chunk_elapsed  = batch_wall_t1 - chunk_wall_t0
        batch_rate     = _fmt_rate(n_batch,    batch_elapsed)
        chunk_rate     = _fmt_rate(total_rows, chunk_elapsed)
        n_active_B     = len(active_std_vids["B"])
        n_active_A     = len(active_std_vids["A"])
        n_unres        = len(_unresolved_std) + len(_unresolved_shd)

        logger.info(
            "  [%s] batch %4d  rows=%6d  batch=%.2fs(%s)  chunk=%.1fs(%s)"
            "  | placed=%3d  B=%3d  A=%3d"
            "  | fills=%5d  cxls=%5d  shd=%5d"
            "  | unres=%4d  total=%8d",
            chunk_label,
            batch_idx,
            n_batch,
            batch_elapsed,
            batch_rate,
            chunk_elapsed,
            chunk_rate,
            n_placed_batch,
            n_active_B,
            n_active_A,
            n_std_fills,
            n_std_cxls,
            n_shd_fills,
            n_unres,
            total_rows,
        )
        del n_placed_batch  # log-only; reassigned fresh each batch

    # ── Final mid back-fill pass ──────────────────────────────────────────
    # Resolves any events whose +100ms window extended beyond the last batch.
    _fill_mid_unresolved(prev_ts_ns)
    n_still_unresolved = len(_unresolved_std) + len(_unresolved_shd)

    chunk_elapsed = _time.monotonic() - chunk_wall_t0
    logger.info(
        "╚══ Chunk %s done ══╝  total_rows=%d  placed=%d  "
        "std_fills=%d  std_cxls=%d  shd_fills=%d  "
        "unresolved_mid=%d  sessions=%d  warmup_rows_skipped=%d  "
        "elapsed=%.1fs",
        chunk_label, total_rows, n_placed,
        n_std_fills, n_std_cxls, n_shd_fills,
        n_still_unresolved, current_sid + 1, n_warmup_skips,
        chunk_elapsed,
    )
    if n_still_unresolved > 0:
        logger.warning(
            "  %d event(s) still have NaN mid after final pass — "
            "data ends before +100ms window closes (normal at EOD).",
            n_still_unresolved,
        )

    terminal_std = [e for e in std_events if e.get("terminal")]
    if not terminal_std:
        logger.warning("Chunk %s: no terminal standard events — skipping Stage 3.", chunk_label)
        del std_events, shd_events, terminal_std
        return

    # ── Lifetime summary outputs ──────────────────────────────────────────
    # ── Stage 3: contribution calculation ────────────────────────────────
    logger.info(
        "  Stage 3: %d terminal std events / %d terminal shd events → ContributionEngine",
        len(terminal_std),
        sum(1 for e in shd_events if e.get("terminal")),
    )
    engine      = ContributionEngine(tick_size=tick_size)
    fill_rows   = engine.calculate(std_events, shd_events)
    cancel_rows = engine.calculate_cancel_value(std_events, shd_events)

    # ── Back-populate reward into lifetime.completed ──────────────────────
    # lifetime.record() fires per-event during the hot loop, BEFORE the
    # ContributionEngine runs.  At that point t["reward"]=NaN (set in
    # _make_event).  The engine writes total_ticks back onto the event dict
    # but lifetime.completed is already frozen.  We fix that here by building
    # a vid→total_ticks lookup from fill_rows and patching each completed
    # entry whose terminal_trigger is FILL.  This makes episode_return and
    # therefore sharpe_by_queue non-NaN.
    if fill_rows:
        # Build a richer reward map: total_ticks, toxic_ticks, reposition_ticks, case
        _reward_map = {}
        for r in fill_rows:
            vid = r.get("virtual_id", "")
            if not vid:
                continue
            tot  = r.get("total_ticks",       np.nan)
            tox  = r.get("toxic_ticks",        np.nan)
            repo = r.get("reposition_ticks",   np.nan)
            case = r.get("reposition_case",    "")
            if tot == tot:  # skip NaN total
                _reward_map[vid] = (tot, tox, repo, case)

        _n_patched = 0
        for lt in lifetime.completed:
            if lt.get("terminal_trigger") == TRIG_FILL:
                vid = lt.get("virtual_id", "")
                if vid in _reward_map:
                    tot, tox, repo, case = _reward_map[vid]
                    lt["episode_return"]   = tot
                    lt["toxic_ticks"]      = tox
                    lt["reposition_ticks"] = repo
                    lt["reposition_case"]  = case
                    _n_patched += 1
        logger.info(
            "  Reward back-population: patched %d / %d fill entries in lifetime",
            _n_patched, sum(1 for lt in lifetime.completed
                            if lt.get("terminal_trigger") == TRIG_FILL),
        )
        del _reward_map

    # Save lifetime AFTER reward back-population so sharpe_by_queue is valid
    _save_lifetime_summary(lifetime, out, save_prefix, chunk_label)
    del lifetime

    del std_events, shd_events, terminal_std, engine

    logger.info(
        "  Stage 3 done: %d fill rows  %d cancel rows",
        len(fill_rows), len(cancel_rows),
    )

    # ── Save per-chunk CSVs ───────────────────────────────────────────────
    if fill_rows:
        fill_path = out / f"{save_prefix}_{chunk_label}_contributions_raw.csv"
        pd.DataFrame(fill_rows).drop(columns=["state"], errors="ignore").to_csv(
            fill_path, index=False)
        logger.info("  Saved fill contributions → %s  (%d rows)", fill_path, len(fill_rows))
    else:
        logger.warning("  Chunk %s: no fill contribution rows.", chunk_label)
    del fill_rows

    if cancel_rows:
        cancel_path = out / f"{save_prefix}_{chunk_label}_cancel_value_raw.csv"
        pd.DataFrame(cancel_rows).to_csv(cancel_path, index=False)
        logger.info("  Saved cancel values      → %s  (%d rows)", cancel_path, len(cancel_rows))
    del cancel_rows

    # Per-day CSVs written above; all large objects freed.


def main(
    raw_glob:       str,
    instrument_id:  int,
    workdir:        str,
    outdir:         str,
    save_prefix:    str,
    tick_size:      float,
    date:           str  = "",
    warmup_minutes: int  = 10,
    days_per_chunk: int  = 1,
    batch_size:     int  = 100_000,
) -> None:
    """
    Chunked pipeline.

    The full MBO parquet is split into slices of `days_per_chunk` calendar
    days.  Each chunk is processed independently (Stage 1-3), producing a
    list of contribution rows.  After all chunks are done, the rows are
    concatenated and Stages 4 (summary statistics) runs once over the full
    combined set.

    Memory high-water mark
    ----------------------
    At any point only ONE chunk's messages_df + events + contribution rows
    are live.  Previous chunks are reduced to their small contribution-row
    lists before the next chunk loads.

    date : optional YYYYMMDD string (e.g. 20260206)
        When provided the intermediate parquet is named:
            instrument_{id}_{date}_multi_day_data.parquet
        so repeated single-day runs never overwrite each other.
        When omitted, falls back to the legacy name (no date suffix),
        preserving backward compatibility with multi-day runs.

    Combining chunks
    ----------------
    Contribution rows are flat dicts keyed by virtual_id.  They have no
    cross-chunk dependencies — fills from day 1 and day 5 are independent
    observations and can be concatenated directly.  The final groupby in
    build_level_summary / build_date_level_summary handles them uniformly.
    """
    root      = Path(workdir)
    out       = Path(outdir)
    data_proc = root / "data" / "processed"
    data_proc.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Build parquet filename: include date suffix when --date is given so
    # each day's source parquet is stored separately and never overwritten.
    # Format: instrument_{id}_{YYYYMMDD}_multi_day_data.parquet        (e.g. instrument_7152_20260206_multi_day_data.parquet)
    #         instrument_{id}_multi_day_data.parquet  (legacy, no date)
    _date_tag  = f"_{date}" if date else ""
    multiday_path = data_proc / f"instrument_{instrument_id}{_date_tag}_multi_day_data.parquet"

    # ── Stage 0: build full parquet (once) ────────────────────────────────
    stage0_multiday(raw_glob, instrument_id, multiday_path)

    # ── Load metadata only (timestamps) to build chunk boundaries ─────────
    logger.info("Reading timestamps to build %d-day chunk boundaries …", days_per_chunk)
    ts_series = pl.scan_parquet(str(multiday_path)).select("timestamp").collect()
    ts_ns_all = ts_series["timestamp"].cast(pl.Int64).to_numpy()
    del ts_series

    # Derive calendar date for every row (UTC)
    dates_all = (ts_ns_all // (86_400 * 10**9)).astype(int)   # days since epoch
    del ts_ns_all
    unique_days = sorted(set(dates_all.tolist()))
    del dates_all
    n_days   = len(unique_days)
    logger.info("Dataset spans %d unique calendar days.", n_days)

    # Build chunk boundaries: groups of days_per_chunk days
    chunks = []
    for start in range(0, n_days, days_per_chunk):
        day_group  = unique_days[start : start + days_per_chunk]
        day_min    = day_group[0]
        day_max    = day_group[-1]
        # Label: first day in the chunk (YYYY-MM-DD)
        import datetime
        label = datetime.date.fromordinal(
            datetime.date(1970,1,1).toordinal() + day_min
        ).strftime("%Y-%m-%d")
        chunks.append((label, day_min, day_max))

    logger.info("Processing %d chunk(s) of up to %d day(s) each.",
                len(chunks), days_per_chunk)

    # ── Chunked processing ────────────────────────────────────────────────
    # Each chunk writes its own per-day CSVs independently.
    # No cross-chunk accumulation — use summarise_pipeline.py after all
    # chunks are done to aggregate across days without loading everything
    # into RAM simultaneously.

    n_chunks_done = 0

    for chunk_idx, (label, day_min, day_max) in enumerate(chunks):
        logger.info("Processing chunk %d/%d  [%s]  days %d–%d …",
                    chunk_idx + 1, len(chunks), label, day_min, day_max)

        process_chunk(
            parquet_path   = multiday_path,
            day_min        = day_min,
            day_max        = day_max,
            chunk_label    = label,
            out            = out,
            save_prefix    = save_prefix,
            tick_size      = tick_size,
            warmup_minutes = warmup_minutes,
            seed           = 42 + chunk_idx,
            batch_size     = batch_size,
        )
        n_chunks_done += 1

    logger.info("=" * 60)
    logger.info(
        "Pipeline complete.  %d chunk(s) processed.  Per-day outputs in %s",
        n_chunks_done, out,
    )
    logger.info(
        "Run summarise_pipeline.py --outdir %s --prefix %s  to aggregate.",
        out, save_prefix,
    )
    logger.info("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HFT Counterfactual Contribution Analysis Pipeline",
    )
    parser.add_argument("--raw-glob",        required=True,
                        help="Glob for raw MBO CSV files")
    parser.add_argument("--instrument-id",   required=True, type=int,
                        help="Instrument ID to filter")
    parser.add_argument("--workdir",         default=".",
                        help="Repo root (default: current dir)")
    parser.add_argument("--outdir",          default="outputs_hft",
                        help="Output directory")
    parser.add_argument("--save-prefix",     default="out_contrib",
                        help="Prefix for all output files")
    parser.add_argument("--tick-size",       type=float, default=0.01,
                        help="Instrument tick size (default: 0.01)")
    parser.add_argument("--warmup-minutes",  type=int, default=10,
                        help="Warm-up minutes before first order placement")
    parser.add_argument("--days-per-chunk",  type=int, default=1,
                        help="Calendar days per processing chunk (default: 1).")
    parser.add_argument("--batch-size",      type=int, default=100_000,
                        help="Rows per streaming batch within each chunk "
                             "(default: 100000). Lower = less RAM, more batches.")
    parser.add_argument("--date",            default="",
                        help="Trading date for this run (YYYYMMDD, e.g. 20260206).  When set, "
                             "the intermediate parquet is saved as "
                             "instrument_{id}_{date}_multi_day_data.parquet so "
                             "repeated single-day runs never overwrite each other.")
    # --gamma and --n-iter intentionally removed (FQI is disabled)

    args = parser.parse_args()
    main(
        raw_glob       = args.raw_glob,
        instrument_id  = args.instrument_id,
        workdir        = args.workdir,
        outdir         = args.outdir,
        save_prefix    = args.save_prefix,
        tick_size      = args.tick_size,
        date           = args.date,
        warmup_minutes = args.warmup_minutes,
        days_per_chunk = args.days_per_chunk,
        batch_size     = args.batch_size,
    )