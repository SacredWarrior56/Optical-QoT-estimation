import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROCESSED_PATH = "data/processed"
SPLIT_PATH = "data/splits"
WINDOW_SIZE = 96  # 1 day of 15-min intervals
TEST_RATIO = 0.2

FEATURE_COLS = ["qfactor", "power", "cd", "pmd"]
TARGET_INDICES = [0, 2]  # qfactor, cd

def create_sequences(array, window_size, target_indices):
    X, y = [], []
    for i in range(window_size, len(array)):
        X.append(array[i - window_size:i])
        y.append(array[i, target_indices])  # [qfactor, cd]
    return np.array(X), np.array(y)

def split_and_save(file_name):
    print(f"\n⏳ Processing: {file_name}")
    df = pd.read_csv(os.path.join(PROCESSED_PATH, file_name))

    # Ensure all required columns exist
    df = df[FEATURE_COLS].dropna().reset_index(drop=True)
    data = df.values

    # Train-test split
    split_idx = int((1 - TEST_RATIO) * len(data))
    train, test = data[:split_idx], data[split_idx:]

    # Scale using only training data
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    # Windowing
    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE, TARGET_INDICES)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE, TARGET_INDICES)

    # Save all arrays
    base = file_name.replace(".csv", "")
    os.makedirs(SPLIT_PATH, exist_ok=True)

    np.save(os.path.join(SPLIT_PATH, f"{base}_X_train.npy"), X_train)
    np.save(os.path.join(SPLIT_PATH, f"{base}_y_train.npy"), y_train)
    np.save(os.path.join(SPLIT_PATH, f"{base}_X_test.npy"), X_test)
    np.save(os.path.join(SPLIT_PATH, f"{base}_y_test.npy"), y_test)

    print(f"✅ Done: {base} | X_train: {X_train.shape} | X_test: {X_test.shape}")

def process_all_channels():
    for file in os.listdir(PROCESSED_PATH):
        if file.endswith(".csv"):
            split_and_save(file)

if __name__ == "__main__":
    process_all_channels()
