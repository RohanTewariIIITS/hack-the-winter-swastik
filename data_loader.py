import polars as pl
from datasets import load_dataset
import os
from pathlib import Path
from config import *

# NOTE:
# This loader supports HuggingFace streaming datasets.
# For Kaggle datasets, convert CSV/JSON to Parquet first for scalability.

class DataLoader:
    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.raw_dir / "submissions.parquet"

    def fetch_and_save(self):
        """
        Fetches the dataset from HuggingFace and saves it locally as a Parquet file.
        Using streaming to handle large datasets and writing in chunks if necessary.
        """
        if self.parquet_path.exists():
            print(f"Data already exists at {self.parquet_path}. Skipping download.")
            return

        print(f"Downloading dataset {DATASET_NAME}...")
        # Load dataset with streaming enabled
        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
        
        import pyarrow as pa
        import pyarrow.parquet as pq

        print("Streaming dataset to Parquet (chunked write)...")

        writer = None
        batch_size = 100_000

        buffer = []
        for idx, row in enumerate(ds):
            buffer.append(row)
            if len(buffer) >= batch_size:
                table = pa.Table.from_pylist(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(self.parquet_path, table.schema)
                writer.write_table(table)
                buffer.clear()

        # Write remaining rows
        if buffer:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(self.parquet_path, table.schema)
            writer.write_table(table)

        if writer:
            writer.close()

        print(f"Saved raw data to {self.parquet_path}")

    def load_raw(self) -> pl.LazyFrame:
        """
        Loads the raw data as a Polars LazyFrame.
        """
        return pl.scan_parquet(str(self.parquet_path))
    
    def clean_and_prepare(self) -> pl.LazyFrame:
        """
        Basic cleaning:
        - Cast types
        - Filter out invalid verdicts if necessary (we keep them for activity features generally, but maybe filter to OK for improvement analysis)
        - Handle null problem ratings (maybe fill or drop)
        """
        lf = self.load_raw()
        
        required_columns = {
            COL_HANDLE,
            COL_TIMESTAMP,
            COL_USER_RATING,
            COL_PROBLEM_ID,
            COL_VERDICT
        }

        missing = required_columns - set(lf.columns)
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")
        
        # Standardize types
        lf = lf.with_columns([
            pl.col(COL_TIMESTAMP).cast(pl.Int64),
            pl.col(COL_USER_RATING).cast(pl.Float64),
            pl.col(COL_PROBLEM_RATING).cast(pl.Float64)
        ])
        
        lf = lf.filter(
            (pl.col(COL_USER_RATING).is_not_null()) &
            (pl.col(COL_USER_RATING) >= MIN_RATING) &
            (pl.col(COL_USER_RATING) <= MAX_RATING)
        )
        
        # Filter rows where user rating is null? Or problem rating is null?
        # For now, let's just expose the lazy frame.
        return lf

if __name__ == "__main__":
    loader = DataLoader()
    loader.fetch_and_save()
    
    # Quick check
    lf = loader.clean_and_prepare()
    print(f"Schema check passed. Showing sample rows:")
    print(lf.head(5).collect())
