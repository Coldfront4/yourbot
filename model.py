import torch
import torch.nn as nn
from logger import get_logger

logger = get_logger("model")

class AdjustmentModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super(AdjustmentModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

adjustment_model = AdjustmentModel()

try:
    adjustment_model.load_state_dict(torch.load("adjustment_model.pth"))
    logger.info("Loaded adjustment model weights from file.")
except FileNotFoundError:
    logger.info("No pretrained adjustment model found; using fresh weights.")
