# ============================================================
# RESULTS (paper-style) — INLINE PLOTS, NUMERIC-SAFE
# Requires: out_uncon, out_con, QPOS_QUINT, SAVE_PREFIX
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def main(prefix: str, outdir: str):
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    figs_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    SAVE_PREFIX = prefix
    QPOS_QUINT = ["top", "top-mid", "mid", "mid-back", "back"]

    out_uncon = pd.read_parquet(f"{SAVE_PREFIX}_qvalues_unconstrained.parquet")
    out_con = pd.read_parquet(f"{SAVE_PREFIX}_qvalues_constrained.parquet")

    # ---------- Helper: build a clean numeric design matrix ----------
    def _design_numeric(y, X_num=None, X_cat=None):
        X_parts = []
        if X_num:
            num = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in X_num.items()})
            X_parts.append(num)
        if X_cat:
            cat_df = pd.DataFrame({k: v.astype("category") for k, v in X_cat.items()})
            dums = pd.get_dummies(cat_df, drop_first=True, dtype=float)
            X_parts.append(dums)

        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=y.index)
        y = pd.to_numeric(y, errors="coerce")

        mask = y.notna()
        if not X.empty:
            arr = X.to_numpy()
            mask &= np.isfinite(arr).all(axis=1)

        y = y.loc[mask]
        X = X.loc[mask]
        if len(y) < 10 or X.shape[1] == 0:
            return None, None

        X = sm.add_constant(X.astype(float), has_constant="add")
        return y, X

    # ========== A) EV by price level ==========
    ev = out_uncon.query("I=='0'")[["L", "Vval"]].copy()
    ev_stats = (ev.groupby("L")["Vval"]
                  .agg(mean="mean", std="std", median="median",
                       q25=lambda x: x.quantile(0.25),
                       q75=lambda x: x.quantile(0.75))
                  .reindex(["0", "1", "2"]))
    print("\n[EV by price level]\n", ev_stats)
    ev_stats.to_csv(tables_dir / f"{Path(SAVE_PREFIX).name}_ev_by_level.csv")

    plt.figure(figsize=(6, 4))
    ev.boxplot(by="L", column="Vval", grid=False)
    plt.title("Value by Price Level (L)")
    plt.suptitle("")
    plt.xlabel("L (0=best, 2=two ticks back)")
    plt.ylabel("Value (EV)")
    plt.tight_layout()
    plt.savefig(figs_dir / "fig_value_by_pricelevel.png", dpi=160)
    plt.close()

    # ========== B) Queue position regression (per L) ==========
    def queuepos_reg(level_label: str):
        d = out_uncon.query("I=='0' and L==@level_label")["Vval Q qB0 qB1 qB2 qA0".split()].copy()
        if d.empty:
            print(f"L={level_label}: no rows for queuepos_reg.")
            return None
        qmap = {q: i / (len(QPOS_QUINT) - 1) for i, q in enumerate(QPOS_QUINT)}
        d["QueuePos"] = d["Q"].map(qmap)

        y, X = _design_numeric(
            y=d["Vval"],
            X_num={"QueuePos": d["QueuePos"]},
            X_cat={"qB0": d["qB0"], "qB1": d["qB1"], "qB2": d["qB2"], "qA0": d["qA0"]},
        )
        if y is None:
            print(f"L={level_label}: not enough clean data for queuepos_reg.")
            return None

        res = sm.OLS(y, X).fit()
        beta = res.params.get("QueuePos", np.nan)
        tval = res.tvalues.get("QueuePos", np.nan)
        print(f"QueuePos effect at L={level_label}: beta={beta:.4f}, t={tval:.2f}  (expect negative, weaker deeper)")
        return res

    rows = []
    for lvl in ["0", "1", "2"]:
        res = queuepos_reg(lvl)
        if res is not None:
            rows.append({"level": lvl,
                         "beta_QueuePos": float(res.params.get("QueuePos", np.nan)),
                         "t_QueuePos":    float(res.tvalues.get("QueuePos", np.nan))})
    queuepos_df = pd.DataFrame(rows)
    if not queuepos_df.empty:
        queuepos_df.to_csv(tables_dir / f"{Path(SAVE_PREFIX).name}_queuepos_effect.csv", index=False)

    # ========== C) Queue size regressions (Eq. 9 proxy) ==========
    def queuesize_regs(level_label: str):
        d = out_uncon.query("I=='0' and L==@level_label")["Vval qB0 qB1 qB2 qA0 Q".split()].copy()
        if d.empty:
            print(f"L={level_label}: no rows for queuesize_regs.")
            return None

        y, X = _design_numeric(
            y=d["Vval"],
            X_cat={"qB0": d["qB0"], "qB1": d["qB1"], "qB2": d["qB2"], "qA0": d["qA0"], "Q": d["Q"]},
        )
        if y is None:
            print(f"L={level_label}: not enough clean data for queuesize_regs.")
            return None

        res = sm.OLS(y, X).fit()
        print(f"\n[Queue-size regression] L={level_label}")
        print(res.summary().tables[1])
        with open(tables_dir / f"{Path(SAVE_PREFIX).name}_queuesize_L{level_label}.txt", "w") as f:
            f.write(res.summary().as_text())
        return res

    _ = [queuesize_regs(lvl) for lvl in ["0", "1", "2"]]

    # ========== D) Volatility effect at best level (Eq. 10 proxy) ==========
    def volatility_effect():
        d = out_uncon.query("I=='0' and L=='0'")[
            "Vval qB0 qB1 qB2 qA0 Q Vcat".split()
        ].copy()
        if d.empty:
            print("No rows for volatility regression.")
            return None
        vmap = {"Low": 0, "Med": 1, "Hi": 2}
        d["VolIdx"] = d["Vcat"].map(vmap)

        y, X = _design_numeric(
            y=d["Vval"],
            X_num={"VolIdx": d["VolIdx"]},
            X_cat={"qB0": d["qB0"], "qB1": d["qB1"], "qB2": d["qB2"], "qA0": d["qA0"], "Q": d["Q"]},
        )
        if y is None:
            print("Not enough clean data for volatility regression.")
            return None

        res = sm.OLS(y, X).fit()
        print("\n[Volatility effect at best level]")
        print(res.summary().tables[1])
        return res

    _ = volatility_effect()

    # ========== E) MDA importance (permutation) ==========
    rng = np.random.default_rng(0)

    # Base dataset (only working states I='0')
    _base = out_uncon.query("I=='0'")["Vval L Q qB0 qB1 qB2 qA0".split()].copy()
    if _base.empty:
        print("No rows for MDA.")
    else:
        print("\n[MDA] unique levels per variable:")
        for k in ["L", "Q", "qB0", "qB1", "qB2", "qA0"]:
            print(f"  {k}: {_base[k].nunique()} level(s)")

        def _build_X(df_cat: pd.DataFrame) -> pd.DataFrame:
            X = pd.get_dummies(df_cat, drop_first=True, dtype=float)
            return sm.add_constant(X, has_constant="add")

        y = _base["Vval"].to_numpy()
        X0 = _build_X(_base[["L", "Q", "qB0", "qB1", "qB2", "qA0"]])
        base_fit = sm.OLS(y, X0).fit()

        def _r2_for(df_cat: pd.DataFrame) -> float:
            X = _build_X(df_cat)
            fit = sm.OLS(y, X).fit()
            return fit.rsquared

        base_r2 = base_fit.rsquared
        print(f"[MDA] base R^2: {base_r2:.6f}")

        vars_k = ["L", "Q", "qB0", "qB1", "qB2", "qA0"]
        results = []

        nrep = 50
        for k in vars_k:
            if _base[k].nunique() <= 1:
                results.append({"variable": k, "MDA_mean": 0.0, "MDA_std": 0.0})
                continue
            drops = []
            for _ in range(nrep):
                dfp = _base.copy()
                dfp[k] = rng.permutation(dfp[k].values)
                r2 = _r2_for(dfp[["L", "Q", "qB0", "qB1", "qB2", "qA0"]])
                drops.append(max(0.0, base_r2 - r2))
            results.append({
                "variable": k,
                "MDA_mean": float(np.mean(drops)),
                "MDA_std":  float(np.std(drops)),
            })

        mda_df = pd.DataFrame(results).sort_values("MDA_mean", ascending=False)
        print("\n[Permutation importance (refit OLS, ΔR^2)]")
        print(mda_df.to_string(index=False))
        mda_df.to_csv(tables_dir / f"{Path(SAVE_PREFIX).name}_mda_importance.csv", index=False)

    # ========== F) Option to cancel (unconstrained vs constrained) ==========
    u = out_uncon.query("I=='0'")[
        "state_id Vval L Q qB0 qB1 qB2 qA0".split()
    ].copy()
    c = out_con.query("I=='0'")[["state_id", "Vval"]].rename(columns={"Vval": "V_constr"})
    opt = u.merge(c, on="state_id", how="left")
    opt["OptCancel"] = opt["Vval"] - opt["V_constr"]

    overall = opt["OptCancel"].agg(["mean", "median"])
    byL = opt.groupby("L")["OptCancel"].agg(["mean", "median"]).reindex(["0", "1", "2"])
    best_share = (opt.query("L=='0'")["OptCancel"] / (opt.query("L=='0'")["Vval"] + 1e-9)).mean()

    print("\n[Option to cancel]")
    print("Overall:\n", overall)
    print("\nBy L:\n", byL)
    print(f"\nBest level mean share of value from cancel option ≈ {best_share:.3%}")

    byL.to_csv(tables_dir / f"{Path(SAVE_PREFIX).name}_option_to_cancel_byL.csv")

    plt.figure(figsize=(5, 3.2))
    byL["mean"].plot(kind="bar")
    plt.title("Option-to-Cancel Value by Level")
    plt.xlabel("L")
    plt.ylabel("Mean ΔV (Unconstrained − Constrained)")
    plt.tight_layout()
    plt.savefig(figs_dir / "fig_option_to_cancel_by_L.png", dpi=160)
    plt.close()

    print("\n=== Results saved (CSV/TXT); figures saved to disk ===")

    # --- Additional figures/tables section (preserving original analyses) ---
    # (prefix name not used further)
    uncon_path = f"{SAVE_PREFIX}_qvalues_unconstrained.parquet"
    con_path = f"{SAVE_PREFIX}_qvalues_constrained.parquet"

    df_u = pd.read_parquet(uncon_path)
    df_c = pd.read_parquet(con_path)

    work_u = df_u[df_u['I'] == '0'].copy()
    work_c = df_c[df_c['I'] == '0'].copy()

    Q_ORDER = ['top', 'top-mid', 'mid', 'mid-back', 'back']
    QB_ORDER = ['ESh', 'Sh', 'No', 'Lo', 'ELo']
    VCAT_ORDER = ['Low', 'Med', 'Hi']
    L_ORDER = ['0', '1', '2']

    for col, order in [
        ('Q', Q_ORDER),
        ('qB0', QB_ORDER),
        ('qB1', QB_ORDER),
        ('qB2', ['Short', 'Med', 'Long']),
        ('qA0', QB_ORDER),
        ('V', VCAT_ORDER),
        ('L', L_ORDER),
    ]:
        if col in work_u.columns:
            work_u[col] = pd.Categorical(work_u[col].astype(str), categories=order, ordered=True)
        if col in work_c.columns:
            work_c[col] = pd.Categorical(work_c[col].astype(str), categories=order, ordered=True)

    m = work_u.merge(
        work_c[['state_id', 'Vval']].rename(columns={'Vval': 'V_con'}),
        on='state_id',
        how='left',
    )
    m['opt_cancel'] = (m['Vval'] - m['V_con']).astype(float)

    plt.figure(figsize=(6, 4))
    data = [
        m.loc[m['L'] == '0', 'Vval'],
        m.loc[m['L'] == '1', 'Vval'],
        m.loc[m['L'] == '2', 'Vval'],
    ]
    plt.boxplot(data, labels=['Best bid (L=0)', 'L=1', 'L=2'], showfliers=False)
    plt.ylabel('Expected value (ticks)')
    plt.title('Value of a resting limit order by price level')
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_value_by_pricelevel.png', dpi=160)
    plt.close()

    def q_to_frac(q):
        order = {'top': 0.0, 'top-mid': 0.25, 'mid': 0.5, 'mid-back': 0.75, 'back': 1.0}
        return q.map(order)

    rows = []
    for L in ['0', '1', '2']:
        sub = m[m['L'] == L].copy()
        if sub.empty:
            rows.append({'L': L, 'beta_qpos': np.nan, 'n': 0})
            continue
        sub['qfrac'] = q_to_frac(sub['Q'].astype(str))
        fe_cols_all = ['qB0', 'qB1', 'qB2', 'qA0', 'Vcat']
        use_cols = ['qfrac'] + [c for c in fe_cols_all if c in sub.columns]
        X = pd.get_dummies(sub[use_cols], drop_first=True, dtype=float)
        y = pd.to_numeric(sub['Vval'], errors='coerce').astype('float64')
        mask = (~X.isna().any(axis=1)) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        finite_mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
        X = X.loc[finite_mask]
        y = y.loc[finite_mask]
        X = sm.add_constant(X, has_constant='add')
        try:
            mod = sm.OLS(y, X).fit(cov_type='HC1')
            beta = float(mod.params.get('qfrac', np.nan))
        except Exception:
            beta = np.nan
        rows.append({'L': L, 'beta_qpos': beta, 'n': int(len(y))})

    qpos_df = pd.DataFrame(rows)
    qpos_df.to_csv(tables_dir / 'export_qpos_df.csv', index=False)

    def encode_ord_cat(s):
        if s.dtype.name == 'category':
            return s.cat.codes.astype(float)
        return s.astype('category').cat.codes.astype(float)

    rows = []
    for L in ['0', '1', '2']:
        sub = m[m['L'] == L].copy()
        if sub.empty:
            continue
        sub['qB0_e'] = encode_ord_cat(sub['qB0'])
        sub['qB1_e'] = encode_ord_cat(sub['qB1'])
        sub['qB2_e'] = encode_ord_cat(sub['qB2'])
        sub['qA0_e'] = encode_ord_cat(sub['qA0'])
        X = sub[['qB0_e', 'qB1_e', 'qB2_e', 'qA0_e']].copy()
        X = sm.add_constant(X, has_constant='add')
        y = sub['Vval'].astype(float)
        mod = sm.OLS(y, X).fit(cov_type='HC1')
        rows.append({
            'L': L,
            **{k: mod.params[k] for k in ['qB0_e', 'qB1_e', 'qB2_e', 'qA0_e']}
        })

    qs_df = pd.DataFrame(rows)
    qs_df.to_csv(tables_dir / 'export_qs_df.csv', index=False)

    sub = m[m['L'] == '0'].copy()
    if not sub.empty:
        V_d = pd.get_dummies(sub['Vcat'], drop_first=True, dtype=float)
        ctrl_cols = [c for c in ['qB0', 'qB1', 'qB2', 'qA0', 'Q'] if c in sub.columns]
        C = pd.get_dummies(sub[ctrl_cols], drop_first=True, dtype=float) if ctrl_cols else pd.DataFrame(index=sub.index)
        X = pd.concat([V_d, C], axis=1)
        y = pd.to_numeric(sub['Vval'], errors='coerce').astype('float64')
        X = X.apply(pd.to_numeric, errors='coerce').astype('float64')
        mask = (~X.isna().any(axis=1)) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        finite = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
        X = X.loc[finite]
        y = y.loc[finite]
        X = sm.add_constant(X, has_constant='add')
        mod = sm.OLS(y, X).fit(cov_type='HC1')
        vol_table = pd.DataFrame(mod.params, columns=["coef"])  # save vol coefficients
        vol_table['se'] = mod.bse
        vol_table['t'] = mod.tvalues
        vol_table['p'] = mod.pvalues
        vol_table.to_csv(tables_dir / 'export_volatility_coeffs.csv')

    rng = np.random.default_rng(0)
    vars_k = ['L', 'Q', 'qB0', 'qB1', 'qB2', 'qA0', 'Vcat']
    # _base_idx = m[['state_id', 'Vval']].copy().set_index('state_id')

    def state_key(df):
        return list(zip(
            df['L'].astype(str),
            df['Q'].astype(str),
            df['qB0'].astype(str),
            df['qB1'].astype(str),
            df['qB2'].astype(str),
            df['qA0'].astype(str),
            df['Vcat'].astype(str),
        ))

    key_cols = ['L', 'Q', 'qB0', 'qB1', 'qB2', 'qA0', 'Vcat']
    m_key = m[key_cols + ['Vval']].copy()
    m_key['key'] = state_key(m_key)
    key_to_V = dict(zip(m_key['key'], m_key['Vval']))

    def mda_for(var, n_iter=50):
        diffs = []
        for _ in range(n_iter):
            pert = m[key_cols].copy()
            pert[var] = m[var].sample(frac=1.0, random_state=rng.integers(1e9)).values
            pert['key'] = state_key(pert)
            V_pert = np.array([key_to_V.get(k, np.nan) for k in pert['key']])
            mask = ~np.isnan(V_pert)
            baseV = m.loc[mask, 'Vval'].values
            diff = np.abs(baseV - V_pert[mask]) / np.clip(np.abs(baseV), 1e-9, None)
            diffs.append(diff.mean())
        return float(np.mean(diffs))

    mda_rows = [{'variable': v, 'MDA': mda_for(v)} for v in vars_k]
    mda_df = pd.DataFrame(mda_rows).sort_values('MDA', ascending=False)
    mda_df.to_csv(tables_dir / 'export_mda_df.csv', index=False)

    # Additional plots
    levels = ['0', '1', '2']
    data = [m.loc[m['L'] == L, 'Vval'].astype(float).dropna().values for L in levels]

    def pct_clip(vals, p_lo=1, p_hi=99):
        vals = np.asarray(vals)
        lo, hi = (
            np.nanpercentile(vals[np.isfinite(vals)], [p_lo, p_hi])
            if np.isfinite(vals).any() else (0, 0)
        )
        if lo == hi:
            hi = lo + 1e-6
        return lo, hi
    lo, hi = pct_clip(np.concatenate(data), 1, 99)

    plt.figure(figsize=(7, 4.5))
    v = plt.violinplot(data, positions=range(1, 4), showmeans=False, showextrema=False, widths=0.8)
    for b in v['bodies']:
        b.set_alpha(0.6)
    plt.boxplot(data, positions=range(1, 4), widths=0.35, showfliers=True)
    means = [np.nanmean(d) if len(d) > 0 else np.nan for d in data]
    plt.scatter(range(1, 4), means, zorder=3)
    plt.xticks([1, 2, 3], ['Best bid (L=0)', 'L=1', 'L=2'])
    plt.ylabel('Expected value (ticks)')
    plt.ylim(lo, hi)
    plt.title('Resting limit-order value by price level (tails visible)')
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_value_by_level_violin.png', dpi=160)
    plt.close()

    def ecdf(x):
        x = np.sort(x[np.isfinite(x)])
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    plt.figure(figsize=(7, 4.5))
    for L, label in [('0', 'L=0'), ('1', 'L=1'), ('2', 'L=2')]:
        x = m.loc[m['L'] == L, 'opt_cancel'].astype(float).dropna().values
        if x.size == 0:
            continue
        xs, ys = ecdf(x)
        plt.step(xs, ys, where='post', label=f'{label} ECDF', alpha=0.85)
    plt.xlabel('Δ value from cancel option (ticks)')
    plt.ylabel('Cumulative fraction')
    plt.title('Option-to-cancel — ECDF by price level')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_opt_cancel_ecdf.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for L, label in [('0', 'L=0'), ('1', 'L=1'), ('2', 'L=2')]:
        x = m.loc[m['L'] == L, 'opt_cancel'].astype(float).dropna().values
        if x.size == 0:
            continue
        xs, ys = ecdf(x)
        ccdf = 1 - ys
        plt.step(xs, ccdf, where='post', label=label, alpha=0.9)
    plt.yscale('log')
    plt.xlabel('Δ value from cancel option (ticks)')
    plt.ylabel('Tail probability (1 - CDF, log scale)')
    plt.title('Option-to-cancel — tail behavior by price level')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_opt_cancel_ccdf_log.png', dpi=160)
    plt.close()

    sub = m[m['L'] == '0'].copy()
    order = {'top': 0.0, 'top-mid': 0.25, 'mid': 0.5, 'mid-back': 0.75, 'back': 1.0}
    sub['qfrac'] = sub['Q'].astype(str).map(order)

    def binned_curve(df, xcol, ycol, n_bins=10):
        df = df[[xcol, ycol]].dropna().sort_values(xcol)
        bins = np.linspace(0, 1, n_bins + 1)
        out = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            sl = df[(df[xcol] >= lo) & (df[xcol] < hi)]
            if len(sl) == 0:
                continue
            y = sl[ycol].values.astype(float)
            mu = np.nanmean(y)
            se = np.nanstd(y, ddof=1) / np.sqrt(max(1, len(y)))
            out.append((((lo + hi) / 2), mu, 1.96 * se, len(y)))
        return pd.DataFrame(out, columns=['x', 'mean', 'ci', 'n'])

    curve = binned_curve(sub, 'qfrac', 'Vval', n_bins=10)
    plt.figure(figsize=(7, 4))
    plt.plot(curve['x'], curve['mean'])
    plt.fill_between(curve['x'], curve['mean'] - curve['ci'], curve['mean'] + curve['ci'], alpha=0.2)
    plt.xlabel('Normalized queue fraction (front → back)')
    plt.ylabel('Expected value (ticks)')
    plt.title('Queue-position effect at the touch (binned mean ±95% CI)')
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_qpos_binned_ci.png', dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results tables and figures from qvalues parquet files.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix to qvalues files without suffix, e.g. outputs/qvalues/out_monthly",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory root for tables/ and figures/",
    )
    args = parser.parse_args()
    main(args.prefix, args.outdir)

