import torch
import torch.nn as nn


def count_parameters(model):
    """
    Returns the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Prints model architecture and parameter count.
    """
    print(model)
    print(f"\nTotal Trainable Parameters: {count_parameters(model):,}\n")


def initialize_weights(model, init_type='kaiming'):
    """
    Initialize weights for all layers in a model.
    Supported types: 'kaiming', 'xavier', 'normal'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError("init_type must be one of: 'kaiming', 'xavier', 'normal'")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def clip_gradients(model, max_norm=1.0):
    """
    Clips gradients to avoid exploding gradients during training.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def freeze_layers(model, layers_to_freeze=[]):
    """
    Freezes specified layers (e.g., for transfer learning or ablation).
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
