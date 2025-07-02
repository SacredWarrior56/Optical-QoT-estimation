import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        # Number of windows possible
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Return window slice of shape (sequence_length, num_features)
        x_seq = self.X[idx : idx + self.sequence_length]
        # Target could be the target at the end of the window (common in forecasting)
        y_seq = self.y[idx + self.sequence_length - 1]
        return x_seq, y_seq

