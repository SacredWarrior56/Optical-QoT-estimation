import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def load_and_clean(file_path):
    # Define expected columns
    cols = ["timestamp", "qfactor", "power", "cd", "pmd"]

    # Load data
    df = pd.read_excel(file_path, header=None)
    df.columns = cols

    # Drop timestamp (not needed for modeling)
    df = df.drop(columns=["timestamp"])

    # Drop missing values
    df = df.dropna()

    # Optional: reset index if needed
    df = df.reset_index(drop=True)

    return df

def process_all_channels():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    for fname in os.listdir(RAW_PATH):
        if fname.endswith(".xlsx"):
            df = load_and_clean(os.path.join(RAW_PATH, fname))
            save_path = os.path.join(PROCESSED_PATH, fname.replace(".xlsx", ".csv"))
            df.to_csv(save_path, index=False)
            print(f"✅ Cleaned: {fname}+cleaned → {save_path}")

if __name__ == "__main__":
    process_all_channels()



