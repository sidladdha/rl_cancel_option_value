# ============================================================
# LOB snapshots  →  RL transitions  →  Value iteration  →  Results
# Works with columns:
#   timestamp, bid_price_1, bid_size_1, bid_price_2, bid_size_2,
#   bid_price_3, bid_size_3, ask_price_1, ask_size_1
# ============================================================

# --------------- Imports ----------------
import numpy as np
import pandas as pd
import warnings
from scipy import sparse as sp
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIG (CLI-provided; see main()) ----------------
# Defaults kept for backward compatibility; overridden by CLI args.
PARQUET_PATH = "instrument_38_lob_snapshots.parquet"
DT_IN_ROWS = 1
SAVE_PREFIX = "out_monthly"

# Column names (change if yours differ)
COL = dict(
    ts="timestamp",
    bid_p0="bid_price_1",
    bid_s0="bid_size_1",
    bid_p1="bid_price_2",
    bid_s1="bid_size_2",
    bid_p2="bid_price_3",
    bid_s2="bid_size_3",
    ask_p0="ask_price_1",
    ask_s0="ask_size_1",
)

# State encodings
QUEUE_QUINT = ["ESh", "Sh", "No", "Lo", "ELo"]  # quintiles (extra short → extra long)
QB2_TERC = ["Short", "Med", "Long"]               # terciles
VOL_TERC = ["Low", "Med", "Hi"]
QPOS_QUINT = ["top", "top-mid", "mid", "mid-back", "back"]
ABSORB_TOK = "X"

# Value iteration
GAMMA = 0.99
VTOL = 1e-6
MAX_ITR = 500


# --------------- Helpers ----------------
def qbucket_strict(x, q=None, k=None, labels=None):
    """
    Robust quantile bucketer:
      - if q provided (quantiles in [0,1]), use those; else if k provided, make k equal-prob bins
      - ensures strictly increasing edges (nudges ties)
      - trims/extends labels to match number of bins
    Returns pandas.Categorical (strings).
    """
    s = pd.Series(x, copy=False).astype(float)
    vals = s.dropna().to_numpy()
    if vals.size == 0:
        single = labels[0] if labels else "B0"
        return pd.Categorical([single] * len(s), categories=[single])

    if q is None:
        k = int(k or 5)
        q = np.linspace(0, 1, k + 1)
    edges = np.quantile(vals, q)

    # strictly increasing edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    # degenerate: make tiny span
    if not np.all(np.diff(edges) > 0):
        lo = np.nanmin(vals)
        hi = lo + 1e-12
        edges = np.linspace(lo, hi, num=max(2, (len(labels) + 1) if labels else 2))

    n_bins = len(edges) - 1
    if labels is None:
        use_labels = [f"B{i}" for i in range(n_bins)]
    else:
        if len(labels) >= n_bins:
            use_labels = list(labels[:n_bins])
        else:
            use_labels = list(labels) + [f"B{i}" for i in range(len(labels), n_bins)]

    return pd.cut(s, bins=edges, labels=use_labels, include_lowest=True)


def estimate_tick(spread_series):
    s = pd.Series(spread_series).replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty:
        return 1e-6
    return float(s.median())


def roll_std(x, win):
    a = pd.Series(x)
    return a.rolling(win, min_periods=max(5, win // 4)).std().to_numpy()


def main(snapshots_path: str, dt_in_rows: int, save_prefix: str, outdir: str):
    # --------------- 0) Load & featureize LOB ----------------
    df = pd.read_parquet(snapshots_path)
    need = [
        COL["ts"], COL["bid_p0"], COL["bid_s0"], COL["bid_p1"], COL["bid_s1"],
        COL["bid_p2"], COL["bid_s2"], COL["ask_p0"], COL["ask_s0"]
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = df.sort_values(COL["ts"]).reset_index(drop=True)

    df["mid"] = 0.5 * (df[COL["bid_p0"]].astype(float) + df[COL["ask_p0"]].astype(float))
    df["spread"] = (df[COL["ask_p0"]].astype(float) - df[COL["bid_p0"]].astype(float))
    df["lret"] = np.log(df["mid"]).diff()
    win = max(10, 60)
    df["vol_roll"] = roll_std(df["lret"].fillna(0), win)

    # Volatility terciles
    df["Vcat"] = qbucket_strict(
        df["vol_roll"].fillna(df["vol_roll"].median()), k=3, labels=VOL_TERC
    ).astype(str)

    # Depth buckets (cross-sectional)
    df["qB0"] = qbucket_strict(df[COL["bid_s0"]], k=5, labels=QUEUE_QUINT).astype(str)
    df["qB1"] = qbucket_strict(df[COL["bid_s1"]], k=5, labels=QUEUE_QUINT).astype(str)
    df["qB2"] = qbucket_strict(df[COL["bid_s2"]], k=3, labels=QB2_TERC).astype(str)
    df["qA0"] = qbucket_strict(df[COL["ask_s0"]], k=5, labels=QUEUE_QUINT).astype(str)

    tick = max(estimate_tick(df["spread"]), 1e-6)

    # --------------- 1) Build state map ----------------
    rows = []
    for qB0 in QUEUE_QUINT:
        for qB1 in QUEUE_QUINT:
            for qB2 in QB2_TERC:
                for qA0 in QUEUE_QUINT:
                    for Vc in VOL_TERC:
                        for Lc in ["0", "1", "2"]:
                            for Qc in QPOS_QUINT:
                                rows.append(("0", Lc, Qc, qB0, qB1, qB2, qA0, Vc))
    for qB0 in QUEUE_QUINT:
        for qB1 in QUEUE_QUINT:
            for qB2 in QB2_TERC:
                for qA0 in QUEUE_QUINT:
                    for Vc in VOL_TERC:
                        rows.append(("1", ABSORB_TOK, ABSORB_TOK, qB0, qB1, qB2, qA0, Vc))
    rows.append((ABSORB_TOK,) * 8)

    state_map = pd.DataFrame(rows, columns=["I", "L", "Q", "qB0", "qB1", "qB2", "qA0", "Vcat"])
    state_map = state_map.reset_index().rename(columns={"index": "state_id"})
    ABSORB_ID = int(state_map.query("I=='X' and L=='X' and Q=='X'")["state_id"].iloc[0])
    S = int(state_map.shape[0])

    key_cols = ["I", "L", "Q", "qB0", "qB1", "qB2", "qA0", "Vcat"]

    # --------------- 2) Expand snapshots into start states ----------------
    base_cols = [
        COL["ts"], "mid", "spread", "Vcat", "qB0", "qB1", "qB2", "qA0",
        COL["bid_p0"], COL["bid_s0"], COL["bid_p1"], COL["bid_s1"],
        COL["bid_p2"], COL["bid_s2"], COL["ask_p0"], COL["ask_s0"]
    ]
    base = df[base_cols].copy()

    expanded = []
    for Lc in ["0", "1", "2"]:
        for Qc in QPOS_QUINT:
            tmp = base.copy()
            tmp["I"] = "0"
            tmp["L"] = Lc
            tmp["Q"] = Qc
            expanded.append(tmp)
    states = pd.concat(expanded, ignore_index=True)

    # enforce strings
    for c in ["I", "L", "Q", "qB0", "qB1", "qB2", "qA0", "Vcat"]:
        states[c] = states[c].astype(str)

    # attach start ids, with diagnostics + drop unmapped
    merged = states[["I", "L", "Q", "qB0", "qB1", "qB2", "qA0", "Vcat"]].merge(
        state_map[key_cols + ["state_id"]], on=key_cols, how="left"
    )
    missing_mask = merged["state_id"].isna()
    if missing_mask.any():
        n_bad = int(missing_mask.sum())
        print(f"[WARN] {n_bad} start states didn’t map. showing first 5:")
        print(merged.loc[missing_mask, key_cols].head())
        keep = ~missing_mask
        states = states.loc[keep].reset_index(drop=True)
        merged = merged.loc[keep].reset_index(drop=True)
    state_id_start = merged["state_id"].astype(np.int64).to_numpy()

    # --------------- 3) Next-step alignment ----------------
    states_next = states.shift(-dt_in_rows)
    valid_mask = states_next[COL["ts"]].notna().to_numpy()
    curr = states.loc[valid_mask].reset_index(drop=True)
    nxt = states_next.loc[valid_mask].reset_index(drop=True)
    state_id_start = state_id_start[valid_mask]

    # --------------- 4) Exec detection + forced cancel ----------------
    same_price = (curr[COL["bid_p0"]].values == nxt[COL["bid_p0"]].values)
    depth_drop = (curr[COL["bid_s0"]].values - nxt[COL["bid_s0"]].values)
    exec_vol = np.where(same_price & (depth_drop > 0), depth_drop, 0.0)

    frac_map = {"top": 0.00, "top-mid": 0.125, "mid": 0.375, "mid-back": 0.625, "back": 0.875}
    L_arr = curr["L"].astype(str).values
    Q_arr = curr["Q"].astype(str).values
    lvl_size = curr[COL["bid_s0"]].astype(float).values
    size_ahead = np.array([frac_map[q] * max(l, 1e-9) for q, l in zip(Q_arr, lvl_size)], dtype=float)
    filled_L0 = (L_arr == "0") & (exec_vol >= size_ahead)

    bp0_now = curr[COL["bid_p0"]].astype(float).values
    bp0_nxt = nxt[COL["bid_p0"]].astype(float).values
    dist_ticks = np.round((bp0_nxt - bp0_now) / tick)
    L_int = L_arr.astype(int)
    improved_into_best = dist_ticks < -L_int

    filled = filled_L0 | improved_into_best

    new_L = np.maximum(0, L_int - dist_ticks)
    forced_cancel = new_L > 2

    # --------------- 5) Rewards ----------------
    mid_now = curr["mid"].astype(float).values
    mid_nxt = nxt["mid"].astype(float).values
    I_start = np.zeros(len(curr), dtype=int)
    carry_rew = (mid_nxt - mid_now) * I_start
    exec_price = np.where(filled, np.where(improved_into_best, bp0_nxt, bp0_now), np.nan)
    exec_rew = np.where(filled, (mid_nxt - exec_price), 0.0)
    R_immediate = carry_rew + exec_rew

    # --------------- 6) End state & validity checks ----------------
    state_id_end = np.where(forced_cancel, ABSORB_ID, state_id_start)

    valid_idx = (
        (~pd.isna(state_id_start)) & (~pd.isna(state_id_end))
        & (state_id_start >= 0) & (state_id_end >= 0)
        & (state_id_start < S) & (state_id_end < S)
    )
    if not np.all(valid_idx):
        bad = int((~valid_idx).sum())
        print(f"[WARN] Dropping {bad} invalid transitions.")
        for name, arr in [
            ("state_id_start", state_id_start),
            ("state_id_end", state_id_end),
            ("forced_cancel", forced_cancel),
            ("exec_price", exec_price),
            ("mid_now", mid_now),
            ("mid_nxt", mid_nxt),
            ("I_start", I_start),
            ("R_immediate", R_immediate),
        ]:
            if isinstance(arr, np.ndarray):
                locals()[name] = arr[valid_idx]
        curr = curr.loc[valid_idx].reset_index(drop=True)
        nxt = nxt.loc[valid_idx].reset_index(drop=True)

    # --------------- 7) Transitions parquet ----------------
    transitions = pd.DataFrame({
        "state_id_start": state_id_start.astype(int),
        "state_id_end": state_id_end.astype(int),
        "action_forced_cancel": forced_cancel.astype(bool),
        "exec_price": exec_price,
        "mid_start": mid_now,
        "mid_end": mid_nxt,
        "I_start": I_start,
        "R_immediate": R_immediate,
        "I": curr["I"].astype(str).values,
        "L": curr["L"].astype(str).values,
        "Q": curr["Q"].astype(str).values,
        "qB0": curr["qB0"].astype(str).values,
        "qB1": curr["qB1"].astype(str).values,
        "qB2": curr["qB2"].astype(str).values,
        "qA0": curr["qA0"].astype(str).values,
        "Vcat": curr["Vcat"].astype(str).values,
    })

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    transitions_path = out_path / f"{save_prefix}_rl_transitions.parquet"
    transitions.to_parquet(str(transitions_path), index=False)
    print(f"Wrote {transitions_path}  (rows={len(transitions):,})")

    # ============================================================
    # VALUE ITERATION (actions: NA, CANCEL)
    # ============================================================
    S = int(state_map.shape[0])

    # Empirical T_NA and expected R_NA
    grp = transitions.groupby(["state_id_start", "state_id_end"]).size().rename("cnt").reset_index()
    row_sum = grp.groupby("state_id_start")["cnt"].transform("sum")
    probs = grp.assign(p=grp["cnt"] / row_sum)

    T_rows = probs["state_id_start"].to_numpy()
    T_cols = probs["state_id_end"].to_numpy()
    T_vals = probs["p"].to_numpy()

    # Guard
    if (T_rows.min() < 0) or (T_cols.min() < 0) or (T_rows.max() >= S) or (T_cols.max() >= S):
        raise ValueError("Out-of-range indices when building T_NA; check mapping step.")

    T_NA = sp.csr_matrix((T_vals, (T_rows, T_cols)), shape=(S, S))

    rew_grp = (
        transitions.groupby(["state_id_start", "state_id_end"])['R_immediate']
        .mean()
        .rename("r_bar")
        .reset_index()
    )
    rew = probs.merge(rew_grp, on=["state_id_start", "state_id_end"], how="inner")
    R_vals = (rew["p"] * rew["r_bar"]).to_numpy()
    R_rows = rew["state_id_start"].to_numpy()
    R_cols = rew["state_id_end"].to_numpy()
    R_NA = sp.csr_matrix((R_vals, (R_rows, R_cols)), shape=(S, S))

    # Cancel action: from working states (I="0") jump to ABSORB_ID
    mask_work = (state_map["I"].values == "0")
    rows, cols, vals = [], [], []
    for sid, ok in enumerate(mask_work):
        if ok:
            rows.append(sid)
            cols.append(ABSORB_ID)
            vals.append(1.0)
    T_C = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
    R_C = sp.csr_matrix((S, S))

    def value_iteration(Ts, Rs, gamma=GAMMA, tol=VTOL, max_iter=MAX_ITR):
        S_loc = Ts[0].shape[0]
        V = np.zeros(S_loc, dtype=float)
        for _ in range(max_iter):
            Qs = []
            for a in range(len(Ts)):
                q_a = Rs[a].sum(axis=1).A.ravel() + gamma * Ts[a].dot(V)
                Qs.append(q_a)
            V_new = Qs[0] if len(Qs) == 1 else np.maximum.reduce(Qs)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new, Qs
            V = V_new
        return V, Qs

    # Solve
    V_uncon, Qs_u = value_iteration([T_NA, T_C], [R_NA, R_C])
    V_con, Qs_c = value_iteration([T_NA], [R_NA])

    # Collect
    out_uncon = state_map.copy()
    out_uncon["Vval"] = V_uncon
    out_uncon["Q_NA"] = Qs_u[0]
    out_uncon["Q_C"] = Qs_u[1]

    out_con = state_map.copy()
    out_con["Vval"] = V_con

    qvals_uncon = out_path / f"{save_prefix}_qvalues_unconstrained.parquet"
    qvals_con = out_path / f"{save_prefix}_qvalues_constrained.parquet"
    out_uncon.to_parquet(str(qvals_uncon), index=False)
    out_con.to_parquet(str(qvals_con), index=False)
    print(f"Saved q-values: {qvals_uncon}, {qvals_con}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RL transitions and run value iteration.")
    parser.add_argument(
        "--snapshots",
        required=False,
        default=PARQUET_PATH,
        help="Path to LOB snapshots parquet (input)",
    )
    parser.add_argument(
        "--dt-in-rows",
        type=int,
        default=DT_IN_ROWS,
        help="Row shift to define next-step alignment",
    )
    parser.add_argument(
        "--save-prefix",
        default=SAVE_PREFIX,
        help="Prefix for output parquet filenames",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/qvalues",
        help="Directory for output parquet files",
    )
    args = parser.parse_args()

    main(
        snapshots_path=args.snapshots,
        dt_in_rows=args.dt_in_rows,
        save_prefix=args.save_prefix,
        outdir=args.outdir,
    )