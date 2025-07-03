import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# === CONFIG ===
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/correlation_analysis"
FEATURES = ['qfactor', 'power', 'cd', 'pmd']

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(DATA_DIR, file)
    print(f"\n🔍 Processing file: {file}")

    df = pd.read_csv(file_path)

    if not all(col in df.columns for col in FEATURES):
        print(f"⚠️  Skipping {file} — missing required features.")
        continue

    df = df[FEATURES]
    corr_matrix = df.corr()
    pmd_q_corr = corr_matrix.loc['qfactor', 'pmd']
    print(f"📌 Correlation (PMD vs Q-Factor): {pmd_q_corr:.4f}")

    # 🔤 Clean filename into a consistent identifier (e.g., channel_97_segment_3)
    base_name = file.lower().replace(",", "").replace(" ", "_").replace(".csv", "")

    # 🎨 Plot + save
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix - {file}")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()

print("\n✅ Done with all files!")
