import os
import pandas as pd
import argparse
from typing import Tuple
from sklearn.model_selection import train_test_split

import pyarrow as pa
import pyarrow.parquet as pq

def detect_columns_from_schema(schema: pa.Schema) -> Tuple[str, str, str, str]:
    """Auto-detects columns from the Parquet file schema."""
    column_names = schema.names
    id_pairs = [("Src IP", "Dst IP"), ("src_ip", "dst_ip")]
    ts_candidates = ["Timestamp", "timestamp", "Time", "time", "ts"]
    label_candidates = ["target", "label", "is_malicious", "malicious"]
    
    src_col, dst_col, ts_col, label_col = None, None, None, None
    
    for s, d in id_pairs:
        if s in column_names and d in column_names:
            src_col, dst_col = s, d
            break
    if src_col is None: raise ValueError("Could not detect source/destination IP columns.")
        
    for t in ts_candidates:
        if t in column_names:
            ts_col = t
            break
    if ts_col is None: raise ValueError("Could not detect a timestamp column.")
        
    for l in label_candidates:
        if l in column_names:
            label_col = l
            break
    if label_col is None: raise ValueError("Could not detect the label column.")
        
    return src_col, dst_col, ts_col, label_col

def create_modified_schema(original_schema: pa.Schema, ts_col: str) -> pa.Schema:
    """Creates a new schema with the timestamp column type changed to timestamp[ns]."""
    fields = []
    for field in original_schema:
        if field.name == ts_col:
            # Replace the old field with a new one that has the correct timestamp type
            fields.append(pa.field(ts_col, pa.timestamp('ns')))
        else:
            fields.append(field)
    # Reconstruct the schema with the updated field list and original metadata
    return pa.schema(fields).with_metadata(original_schema.metadata)


def perform_efficient_temporal_split(input_path: str, ts_col: str, train_ratio: float, output_dir: str, original_schema: pa.Schema):
    """Performs a memory-efficient temporal split with correct schema handling."""
    print(f"Performing temporal split on column: '{ts_col}'")
    
    parquet_file = pq.ParquetFile(input_path)
    
    # Step 1: Find the split point from the timestamp column.
    print("Step 1: Reading timestamp column to determine the split point...")
    timestamps = parquet_file.read(columns=[ts_col]).to_pandas()[ts_col]
    sorted_epochs = pd.to_datetime(timestamps).astype('int64').sort_values()
    split_index = int(len(sorted_epochs) * train_ratio)
    split_timestamp_epoch = sorted_epochs.iloc[split_index]
    split_timestamp = pd.to_datetime(split_timestamp_epoch)
    print(f"Data will be split at timestamp: {split_timestamp}")

    # Step 2: Create the corrected schema for the output files.
    # **THE FIX IS HERE**: We define our desired output schema upfront.
    output_schema = create_modified_schema(original_schema, ts_col)
    
    # Step 3: Iterate and write to new files using the corrected schema.
    print("Step 3: Iterating through dataset and writing split files...")
    train_path = os.path.join(output_dir, "train_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")

    train_writer = None
    test_writer = None

    try:
        # Initialize writers with the NEW, correct output schema.
        train_writer = pq.ParquetWriter(train_path, output_schema)
        test_writer = pq.ParquetWriter(test_path, output_schema)

        for batch in parquet_file.iter_batches(batch_size=200_000, columns=original_schema.names):
            chunk_df = batch.to_pandas()
            # Convert timestamp string to datetime object for comparison.
            chunk_df[ts_col] = pd.to_datetime(chunk_df[ts_col])

            train_chunk = chunk_df[chunk_df[ts_col] < split_timestamp]
            test_chunk = chunk_df[chunk_df[ts_col] >= split_timestamp]

            if not train_chunk.empty:
                # When converting to a table, use the same output_schema. Now they will match.
                train_table = pa.Table.from_pandas(train_chunk, schema=output_schema, preserve_index=False)
                train_writer.write_table(train_table)

            if not test_chunk.empty:
                test_table = pa.Table.from_pandas(test_chunk, schema=output_schema, preserve_index=False)
                test_writer.write_table(test_table)

    finally:
        print("Finalizing write process...")
        if train_writer: train_writer.close()
        if test_writer: test_writer.close()

def perform_stratified_split(input_path: str, label_col: str, train_ratio: float, output_dir: str):
    """Performs a memory-efficient stratified split."""
    print("Loading dataset into memory for stratified split...")
    df = pd.read_parquet(input_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    print(f"Performing stratified split on label column: '{label_col}'")
    
    # Ensure the label column is suitable for stratification
    if df[label_col].nunique() < 2:
        raise ValueError("Label column must have at least two unique values for stratification.")

    train_df, test_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df[label_col],
        random_state=42  # for reproducibility
    )

    train_path = os.path.join(output_dir, "train_set.parquet")
    test_path = os.path.join(output_dir, "test_set.parquet")

    print(f"Writing train set to {train_path}...")
    train_df.to_parquet(train_path, index=False)

    print(f"Writing test set to {test_path}...")
    test_df.to_parquet(test_path, index=False)

    # Print class distribution to verify
    print("\nSplit Verification:")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Train set positive class ratio: {train_df[label_col].mean():.4f}")
    print(f"Test set positive class ratio: {test_df[label_col].mean():.4f}")


def main(args):
    """Main function to orchestrate the splitting process."""
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found at: {args.input}")

    print(f"Inspecting data from: {args.input}")
    
    original_schema = pq.read_schema(args.input)
    _, _, _, label_col = detect_columns_from_schema(original_schema)
    os.makedirs(args.output_dir, exist_ok=True)
    
    perform_stratified_split(args.input, label_col, args.train_ratio, args.output_dir)
    
    print("\nSplit complete!")
    print(f"Train data saved to: {os.path.join(args.output_dir, 'train_set.parquet')}")
    print(f"Test data saved to:  {os.path.join(args.output_dir, 'test_set.parquet')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a memory-efficient temporal split on a Parquet dataset.")
    parser.add_argument("--input", type=str, default="final_dataset.parquet", help="Path to the input Parquet file.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the split files.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of the data for the training set.")
    
    args = parser.parse_args()
    main(args)