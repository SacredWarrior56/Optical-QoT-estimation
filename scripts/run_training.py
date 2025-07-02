import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.train_eval.trainer import TCNTrainer
from src.data_preprocessing.windowing import TimeSeriesDataset
from src.train_eval.metrics import evaluate_regression, print_metrics

# Load config
with open("configs/tcn_config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = "data/processed"

# Loop through all channel files
for file in os.listdir(PROCESSED_PATH):
    if not file.endswith(".csv"):
        continue

    print(f"\n🚀 Starting training for: {file}")

    # Load the data
    data = pd.read_csv(os.path.join(PROCESSED_PATH, file))

    # Features and targets
    features = data[['qfactor', 'power', 'cd']].values
    targets = data[['qfactor', 'cd']].values

    # Train/val split
    train_feat, val_feat, train_tgt, val_tgt = train_test_split(
        features, targets, test_size=0.2, shuffle=False
    )

    # Create dataset objects
    train_dataset = TimeSeriesDataset(train_feat, train_tgt, config['sequence_length'])
    val_dataset = TimeSeriesDataset(val_feat, val_tgt, config['sequence_length'])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Update save path to be per channel
    base_name = file.replace(".csv", "")
    model_path = f"outputs/models/{base_name}_tcn.pth"
    config['save_path'] = model_path

    # Train model
    trainer = TCNTrainer(config, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.fit(num_epochs=config['num_epochs'])

    # Evaluate
    model = trainer.model
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Print final metrics
    metrics = evaluate_regression(all_targets, all_preds)
    print(f"\n📊 Final Metrics for {base_name}")
    print_metrics(metrics, prefix="Validation")
