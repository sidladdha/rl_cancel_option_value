import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd_list):
    print("$", " ".join(cmd_list))
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main(raw_glob: str, instrument_id: int, snapshot_ms: int, workdir: str, results_outdir: str, save_prefix: str):
    root = Path(workdir)
    data_processed = root / "data/processed"
    data_lob = root / "data/lob"
    outputs = Path(results_outdir)
    outputs.mkdir(parents=True, exist_ok=True)
    data_processed.mkdir(parents=True, exist_ok=True)
    data_lob.mkdir(parents=True, exist_ok=True)

    processed_path = data_processed / f"instrument_{instrument_id}_multi_day_data.parquet"
    lob_path = data_lob / f"instrument_{instrument_id}_lob_snapshots.parquet"

    # 1) Build multi-day parquet
    run([
        sys.executable,
        "create_multiday_data.py",
        "--glob", str(raw_glob),
        "--instrument-id", str(instrument_id),
        "--out", str(processed_path),
    ])

    # 2) Build LOB snapshots
    run([
        sys.executable,
        "create_lob_snapshots.py",
        "--input", str(processed_path),
        "--snapshot-interval-ms", str(snapshot_ms),
        "--out", str(lob_path),
    ])

    # 3) Model step (now CLI):
    run([
        sys.executable,
        "model.py",
        "--snapshots", str(lob_path),
        "--dt-in-rows", str(1),
        "--save-prefix", save_prefix,
        "--outdir", str(outputs / "qvalues"),
    ])

    # 4) Results
    # model.py writes qvalue parquets under outputs/qvalues with prefix
    prefix_path = str((outputs / "qvalues") / save_prefix)
    run([
        sys.executable,
        "results.py",
        "--prefix", prefix_path,
        "--outdir", str(outputs),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full RL LOB pipeline end-to-end.")
    parser.add_argument("--raw-glob", required=True, help="Glob of raw CSV files, e.g. /data/xnas-itch-*.mbo.csv")
    parser.add_argument("--instrument-id", type=int, required=True, help="Instrument ID to filter")
    parser.add_argument("--snapshot-interval-ms", type=int, default=100, help="Snapshot interval in milliseconds")
    parser.add_argument("--workdir", default=".", help="Working directory (repo root)")
    parser.add_argument("--outdir", default="outputs", help="Where to put tables/ and figures/")
    parser.add_argument("--save-prefix", default="out_monthly", help="Prefix for qvalue parquet names")
    args = parser.parse_args()

    main(
        raw_glob=args.raw_glob,
        instrument_id=args.instrument_id,
        snapshot_ms=args.snapshot_interval_ms,
        workdir=args.workdir,
        results_outdir=args.outdir,
        save_prefix=args.save_prefix,
    )
