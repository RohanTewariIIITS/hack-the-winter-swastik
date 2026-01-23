from datasets import load_dataset
import polars as pl
import json

def inspect_dataset():
    print("Loading dataset in streaming mode...")
    ds = load_dataset("denkCF/UsersCodeforcesSubmissionsEnd2024", split="train", streaming=True)
    
    print("\nDataset Info:")
    # Attempt to get dataset features if available
    try:
        print(ds.features)
    except Exception as e:
        print(f"Could not print features directly: {e}")

    print("\nFirst 5 relevant items:")
    iterator = iter(ds)
    samples = []
    for _ in range(5):
        try:
            item = next(iterator)
            samples.append(item)
            print(item)
        except StopIteration:
            break
            
    # Convert to polars to check types
    if samples:
        df = pl.from_dicts(samples)
        print("\nPolars Schema inference from samples:")
        print(df.schema)
        
        print("\nSample Dataframe:")
        print(df)

if __name__ == "__main__":
    inspect_dataset()
