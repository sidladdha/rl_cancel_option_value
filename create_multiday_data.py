import polars as pl
import glob
from pathlib import Path
import argparse


def load_multi_day_data(files_glob_pattern: str, instrument_id_to_load: int) -> pl.DataFrame:
    """
    Loads daily trade and order data from multiple CSV files matching a glob pattern,
    filters for a specific instrument ID, and prepares it for order book reconstruction.

    Args:
        files_glob_pattern: A glob pattern to match the input CSV files (e.g., 'data/xnas-*.csv').
        instrument_id_to_load: The numeric instrument ID to filter for.

    Returns:
        A Polars DataFrame containing the cleaned and filtered data from all files.
    """
    # Use glob to find all files that match the specified pattern
    file_paths = glob.glob(files_glob_pattern)
    if not file_paths:
        print(f"Error: No files found matching the pattern '{files_glob_pattern}'.")
        return pl.DataFrame()

    print(f"Found {len(file_paths)} files to process for instrument ID '{instrument_id_to_load}'.")

    # Polars' scan_csv can take a list of files directly and will union them.
    # We let it infer the schema, assuming all files have the same structure.
    print("Scanning all files and inferring schema... (This may take a moment)")
    try:
        lazy_df = pl.scan_csv(file_paths, infer_schema_length=10000, ignore_errors=True)
    except Exception as e:
        print(f"An error occurred during file scanning: {e}")
        return pl.DataFrame()

    print("All files scanned. Now applying filters and transformations.")

    # --- Post-Load Diagnostics ---
    print("\n--- Diagnostics ---")
    filtered_lazy = lazy_df.filter(pl.col('instrument_id') == instrument_id_to_load)

    try:
        # This will trigger a scan of all files to get the count
        after_filter_count = filtered_lazy.select(pl.len()).collect().item()
        print(f"Rows found for instrument ID {instrument_id_to_load} across all files: {after_filter_count}")
        if after_filter_count == 0:
            print("CRITICAL: No rows found for this instrument ID in any of the files.")
            return pl.DataFrame()
    except Exception as e:
        print(f"Error while trying to filter by instrument_id. Is the column name correct? Details: {e}")
        return pl.DataFrame()
    print("--- End Diagnostics ---\n")

    # Chain the rest of the operations on the combined lazy frame
    processed_lazy = filtered_lazy.with_columns([
        pl.col("ts_event").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ", time_unit="ns").alias("timestamp"),
        pl.col("price").alias("price_float"),
        pl.col("size").alias("shares"),
        pl.col("order_id").cast(pl.String)
    ]).filter(
        pl.col('action').is_in(['A', 'C', 'M', 'T', 'F'])
    ).select([
        "timestamp", pl.col("action").alias("message_type"), "side",
        "price_float", "shares", "order_id", "instrument_id",
    ]).sort("timestamp")

    # Collect the final result from all files
    print("Collecting and sorting final DataFrame...")
    df = processed_lazy.collect()
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build multi-day parquet from raw daily CSV message files.")
    parser.add_argument("--glob", required=True, help="Glob for input CSVs, e.g. /data/xnas-itch-*.mbo.csv")
    parser.add_argument("--instrument-id", required=True, type=int, help="Numeric instrument ID to filter")
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output parquet path, e.g. "
            "data/processed/instrument_7152_multi_day_data.parquet"
        ),
    )
    args = parser.parse_args()

    try:
        processed_df = load_multi_day_data(args.glob, args.instrument_id)

        if processed_df.is_empty():
            print(f"Execution finished. No data loaded for instrument ID '{args.instrument_id}'.")
        else:
            print(f"Successfully processed {processed_df.height} records from all files.")
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            processed_df.write_parquet(str(out_path))
            print(f"\nCombined and processed data saved to '{out_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
