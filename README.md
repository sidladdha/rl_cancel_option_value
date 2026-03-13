# HFT Counterfactual Shadow Tracking Pipeline

Measuring the economic value of the embedded cancel option in limit orders — across every queue level, for both ask and bid sides — using a dual virtual order tracking methodology on NASDAQ ITCH MBO data.

---

## What This Does

Every resting limit order carries an embedded option: the trader can cancel before an adverse fill materialises. This pipeline quantifies the dollar value of that option at each of ten queue levels by running two simultaneous virtual orders through every trading day:

- **Standard tracker** — cancel rules are active (QPR signal, 10-tick reposition, 5-min max lifetime)
- **Shadow tracker** — no cancel rules, orders rest until natural fill or session end (the counterfactual)

The difference in outcomes between these two trackers, matched order-by-order, is **V_cancel** — the cancel option value. Across 19 days of data, V_cancel generates **0.5–3.3 additional ticks per order placed** (8–27× the passive fill return), rising monotonically from the touch to level 10. The strategy produces **negative expected value without cancel flexibility** at levels 2 and 3.

---

## Repository Structure

```
.
├── run_hft_pipeline.py        # Main orchestrator — all stages, one or many days
├── virtual_order_tracker.py   # Standard + Shadow order observer (QPR signal, reposition)
├── contribution_engine.py     # V_cancel and fill contribution metrics (in ticks)
├── summarise_pipeline.py      # Cross-day aggregation — produces ALL_* summary files
├── shadow_tracker.py          # Shadow mode helpers
├── reward_engine.py           # (Legacy) Absorbing-state reward engine
├── fqi_model.py               # (Legacy) Fitted Q-Iteration — XGBoost/RF, walk-forward OOS
├── create_multiday_data.py    # Upstream: builds multi-day parquet from raw ITCH CSVs
├── create_lob_snapshots.py    # Upstream: fixed-interval LOB snapshots (original pipeline)
├── model.py                   # Upstream: tabular value iteration (original pipeline)
├── results.py                 # Upstream: results tables and figures (original pipeline)
├── run_all.py                 # Upstream: orchestrates original four-stage pipeline
└── requirements.txt
```

---

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline — one day
python run_hft_pipeline.py \
  --raw-glob       "/data/xnas-itch-*.mbo.csv" \
  --instrument-id  7152 \
  --date           20260206 \
  --workdir        "." \
  --outdir         "outputs_hft" \
  --save-prefix    "out_fqi" \
  --tick-size      0.01 \
  --warmup-minutes 10 \
  --days-per-chunk 1

# Aggregate across days (run after all days are processed)
python summarise_pipeline.py \
  --indir   outputs_hft \
  --outdir  summaries \
  --prefix  out_fqi
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--tick-size` | `0.01` | Instrument tick size in dollars |
| `--warmup-minutes` | `10` | Minutes before first virtual order — lets LOB fully build |
| `--days-per-chunk` | `1` | Days per memory chunk; 1 = lowest memory, recommended |
| `--batch-size` | `100000` | MBO rows per batch |
| `MAX_CONCURRENT` | `100` | Max simultaneous standard+shadow order pairs |
| `BUFFER_NS` | `2,000,000,000` | Mid-price ring buffer lookback (2 seconds) |
| `MAX_LIFETIME_NS` | `300,000,000,000` | Max order lifetime before forced cancel (5 minutes) |
| `QPR_THRESHOLD` | `2.0` | Queue Pressure Ratio cancel trigger |
| `QPR_QFRAC_GATE` | `0.20` | Min queue fraction for QPR cancel to fire |
| `QPR_PERSIST_NS` | `10,000,000` | QPR persistence required before cancel (10ms) |
| `QPR_WINDOW_NS` | `50,000,000` | QPR rolling window (50ms) |
| `REPOSITION_TICKS_THRESHOLD` | `10` | BBO distance in ticks triggering reposition |

---

## Methodology

### Dual Virtual Order Tracking

For each MBO event that represents a new order at levels 1–10, the pipeline spawns a matched standard+shadow pair. Both are exposed to identical market conditions. The standard tracker fires cancel rules; the shadow tracker does not. Terminal events from both are matched and passed to the contribution engine.

### V_cancel Formula

```
V_cancel_ticks = RL_PnL_ticks − Shadow_PnL_ticks

If shadow filled:
  RL_PnL_ticks     = (cancel_mid − shadow_mid_post_100ms) / tick
  Shadow_PnL_ticks = (shadow_fill_price − birth_mid) / tick

If shadow did not fill:
  RL_PnL_ticks     = (ref_mid − birth_mid) / tick
  Shadow_PnL_ticks = 0
```

Positive V_cancel = correct cancel (avoided a loss). Negative V_cancel = premature cancel (price recovered).

### Expected Order Value Decomposition

```
E[V_order]     = P(fill) × E[fill_return] + P(cancel) × E[V_cancel]
E[V_no_option] = P(fill) × E[fill_return] + P(cancel) × E[shadow_PnL]
Option % of EV = P(cancel) × E[V_cancel] / E[V_order]
```

### QPR Cancel Signal (Three-Gate Design)

A cancel fires only when **all three** gates are satisfied simultaneously:
1. **Signal gate**: QPR ≥ 2.0 (self-side depletion vs contra-BBO growth over 50ms)
2. **Depth gate**: q_frac < 0.20 (order is near the front of queue)
3. **Persistence gate**: QPR ≥ 2.0 continuously for ≥ 10ms (filters spoofing/flicker)

---

## Results Summary (19 days, Feb 6 – Mar 5 2026)

### Cancel Option Value and Per-Order Contribution

Cancel contribution/order = P(cancel) × E[V_cancel]. Enhancement ratio = cancel contribution / passive fill return.

| Level | Side | V_cancel (t) | V_cancel (bps) | P(cancel) | Contrib/order (t) | Enhancement ratio | E[V] no-option |
|---|---|---|---|---|---|---|---|
| L1 | Ask | 32.8 | 18.75 | 1.9% | 0.617 | 7.2× | 0.054t |
| L1 | Bid | 30.4 | 17.35 | 1.6% | 0.487 | 7.2× | 0.091t |
| L2 | Ask | 6.5†  | 3.69 | 36.5%† | 2.360 | 19.9× | **−0.330t** ⚠ |
| L2 | Bid | 6.0†  | 3.44 | 35.9%† | 2.157 | 20.6× | **−0.252t** ⚠ |
| L3 | Ask | 16.4 | 9.40 | 11.2% | 1.843 | 15.4× | **−0.072t** ⚠ |
| L3 | Bid | 14.7 | 8.42 | 11.0% | 1.622 | 15.7× | **−0.010t** ⚠ |
| L4 | Ask | 33.3 | 19.04 | 5.3% | 1.778 | 14.5× | 0.012t |
| L4 | Bid | 27.4 | 15.64 | 5.6% | 1.535 | 14.1× | 0.066t |
| L5 | Ask | 44.9 | 25.66 | 4.4% | 1.978 | 16.6× | 0.027t |
| L5 | Bid | 35.8 | 20.43 | 4.7% | 1.678 | 15.0× | 0.088t |
| L6 | Ask | 52.4 | 29.92 | 4.3% | 2.267 | 18.9× | 0.030t |
| L6 | Bid | 40.9 | 23.39 | 4.7% | 1.919 | 17.1× | 0.088t |
| L7 | Ask | 58.2 | 33.27 | 4.6% | 2.662 | 20.7× | 0.041t |
| L7 | Bid | 45.6 | 26.06 | 5.0% | 2.265 | 19.7× | 0.091t |
| L8 | Ask | 62.4 | 35.66 | 4.5% | 2.828 | 22.0× | 0.049t |
| L8 | Bid | 49.0 | 27.98 | 5.0% | 2.448 | 21.9× | 0.079t |
| L9 | Ask | 65.0 | 37.14 | 4.6% | 2.992 | 24.1× | 0.054t |
| L9 | Bid | 53.0 | 30.30 | 5.2% | 2.749 | 24.1× | 0.080t |
| L10 | Ask | 68.1 | 38.90 | 4.9% | 3.316 | 26.3× | 0.061t |
| L10 | Bid | 56.7 | 32.38 | 5.3% | 3.028 | 27.7× | 0.077t |

⚠ = strategy unprofitable without cancel flexibility at this level (existentially necessary, not just value-enhancing).
† = QPR over-trigger artefact at L2; true V_cancel likely higher under recalibrated threshold.

### Reposition Alpha

| Level | Ask | Bid |
|---|---|---|
| L1 | **+654 bps** ✓ | **+585 bps** ✓ |
| L2 | +11 bps (noise) | −29 bps ✗ |
| L3 | +95 bps ✓ | +124 bps ✓ |
| L4 | +120 bps ✓ | −71 bps ✗ |
| L6–L9 | +33–136 bps ✓ | +38–90 bps ✓ |
| L10 | **−85 bps ✗** | **−90 bps ✗** |

---

## Known Issues

| Issue | Priority | Description |
|---|---|---|
| **TUNE-1: L2 QPR over-trigger** | 🔴 High | Cancel rate 36% vs 4–5% elsewhere. Consider level-specific QPR thresholds. |
| **TUNE-2: Bid-side negative repo alpha at L4–L5** | 🟡 Medium | QPR cancels bids during rallies. Investigate side-specific QPR calibration. |
| **TUNE-3: L10 repo rate too high** | 🟡 Medium | 54% reposition rate, −85–90 bps alpha. Suppress reposition at depth ≥ 8. |
| **Case B at L10** | 🟢 Monitor | 4% of L10 fills are max-lifetime cancel refills. |

---

## Reference

> Kwan, A. & Philip, R. (2025). *Reinforcement Learning in a Dynamic Limit Order Market.* NYU Stern Microstructure Meeting 2025.

## License

MIT
