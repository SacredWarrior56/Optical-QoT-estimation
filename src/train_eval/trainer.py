import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.models.tcn import TCN
from src.models.utils import initialize_weights, print_model_summary, clip_gradients


class TCNTrainer:
    def __init__(self, config, train_loader, val_loader, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = TCN(
            num_inputs=config['input_size'],  # Rename input_size to num_inputs
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            output_size=config['output_size']
        ).to(self.device)


        initialize_weights(self.model, init_type='kaiming')
        print_model_summary(self.model)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        self.best_val_loss = float('inf')
        self.save_path = config.get('save_path', 'outputs/models/best_model.pt')

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc='Training', leave=False):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            clip_gradients(self.model, max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', leave=False):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def fit(self, num_epochs=50):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # Save model if validation improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
                print("✅ Model saved!")

        print("\n🏁 Training Complete. Best Val Loss:", self.best_val_loss)


#Part	                        What it does
#fit()	                Trains for n epochs and logs progress
#train_one_epoch()	    One full pass through training data
#validate()	            Evaluates model on validation set
#clip_gradients()	    Prevents unstable training with exploding gradients (from utils.py)
#initialize_weights()	Boosts model convergence (from utils.py)
#outputs/models/	    Saves the best model to this folder#