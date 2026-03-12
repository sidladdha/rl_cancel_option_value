from sortedcontainers import SortedDict
from datetime import timedelta
import polars as pl
from pathlib import Path
import argparse


def reconstruct_and_snapshot_lob(input_parquet_path: str, snapshot_interval_ms: int = 100) -> pl.DataFrame:
    """
    Reconstructs the limit order book from a message stream and creates periodic snapshots.

    Args:
        input_parquet_path: Path to the Parquet file containing processed message data.
        snapshot_interval_ms: The interval in milliseconds for taking snapshots.

    Returns:
        A Polars DataFrame containing time-series snapshots of the LOB.
    """
    print(f"Loading message data from '{input_parquet_path}'...")
    messages_df = pl.read_parquet(input_parquet_path)

    # Initialize order book data structures
    # Bids sorted high-to-low by using a negative key; asks sorted low-to-high.
    bids = SortedDict(lambda k: -k)
    asks = SortedDict()

    # Store individual orders by their ID to allow for modifications and cancellations
    orders = {}

    snapshots = []

    if messages_df.is_empty():
        print("Input DataFrame is empty. Cannot reconstruct LOB.")
        return pl.DataFrame()

    start_time = messages_df['timestamp'][0]
    end_time = messages_df['timestamp'][-1]
    next_snapshot_time = start_time + timedelta(milliseconds=snapshot_interval_ms)

    print(f"Processing {messages_df.height} messages from {start_time} to {end_time}...")

    # Iterate through each message to build the book
    for row in messages_df.iter_rows(named=True):
        timestamp = row['timestamp']
        msg_type = row['message_type']
        order_id = row['order_id']
        side = row['side']
        price = row['price_float']
        shares = row['shares']

        # --- Snapshot Logic ---
        # If the current message's time is past our next snapshot point,
        # take snapshots for all the intervals we've passed.
        while timestamp >= next_snapshot_time:
            # Get the top 3 levels of the book
            bid_prices = list(bids.keys())[:3]
            ask_prices = list(asks.keys())[:3]

            snapshot = {
                'timestamp': next_snapshot_time,
                'bid_price_1': bid_prices[0] if len(bid_prices) > 0 else None,
                'bid_size_1': bids[bid_prices[0]] if len(bid_prices) > 0 else None,
                'bid_price_2': bid_prices[1] if len(bid_prices) > 1 else None,
                'bid_size_2': bids[bid_prices[1]] if len(bid_prices) > 1 else None,
                'bid_price_3': bid_prices[2] if len(bid_prices) > 2 else None,
                'bid_size_3': bids[bid_prices[2]] if len(bid_prices) > 2 else None,
                'ask_price_1': ask_prices[0] if len(ask_prices) > 0 else None,
                'ask_size_1': asks[ask_prices[0]] if len(ask_prices) > 0 else None,
            }
            snapshots.append(snapshot)
            next_snapshot_time += timedelta(milliseconds=snapshot_interval_ms)

        # --- Order Book Logic ---
        if msg_type == 'A':  # Add Order
            if order_id in orders:
                # Duplicate order, ignore
                continue

            # Store order details
            orders[order_id] = {'side': side, 'price': price, 'shares': shares}

            # Update the aggregated book
            book_side = bids if side == 'B' else asks
            book_side[price] = book_side.get(price, 0) + shares

        elif msg_type in ['C', 'M']:
            # Cancel or Modify (treat Modify as Cancel)
            if order_id in orders:
                old_order = orders.pop(order_id)
                book_side = bids if old_order['side'] == 'B' else asks

                # Check if price level exists before trying to subtract
                if old_order['price'] in book_side:
                    book_side[old_order['price']] -= old_order['shares']
                    if book_side[old_order['price']] <= 0:
                        del book_side[old_order['price']]

        elif msg_type == 'F':  # Fill (Execution)
            if order_id in orders:
                order_to_fill = orders[order_id]
                book_side = bids if order_to_fill['side'] == 'B' else asks

                # Check if price level exists
                if order_to_fill['price'] in book_side:
                    book_side[order_to_fill['price']] -= shares
                    if book_side[order_to_fill['price']] <= 0:
                        del book_side[order_to_fill['price']]

                # Update the individual order's remaining shares
                order_to_fill['shares'] -= shares
                if order_to_fill['shares'] <= 0:
                    orders.pop(order_id)

    print(f"Finished processing messages. Created {len(snapshots)} snapshots.")

    if not snapshots:
        return pl.DataFrame()

    # Convert snapshots list to a Polars DataFrame
    snapshot_df = pl.from_dicts(snapshots)

    # Forward-fill any gaps in the snapshots to ensure continuity
    return snapshot_df.sort('timestamp').fill_null(strategy='forward')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct LOB and generate periodic snapshots.")
    parser.add_argument("--input", required=True, help="Input parquet path from multiday step")
    parser.add_argument("--out", required=True, help="Output parquet path for LOB snapshots")
    parser.add_argument("--snapshot-interval-ms", type=int, default=100, help="Snapshot interval in milliseconds")
    args = parser.parse_args()

    try:
        lob_snapshots_df = reconstruct_and_snapshot_lob(args.input, args.snapshot_interval_ms)

        if lob_snapshots_df.is_empty():
            print("Execution finished. No snapshots were generated.")
        else:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            lob_snapshots_df.write_parquet(str(out_path))
            print("\nSuccessfully created LOB snapshots.")
            print(f"Output saved to '{out_path}'.")
            print("\nSnapshot DataFrame sample:")
            print(lob_snapshots_df.head())

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{args.input}'. Please run create_multiday_data.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
