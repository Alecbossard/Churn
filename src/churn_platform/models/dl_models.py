import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple feed-forward network for churn prediction.
    Input: dense feature vector (after preprocessing)
    Output: single logit (binary classification)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output_layer(x)  # shape: (batch, 1)
        return x.squeeze(1)       # shape: (batch,)
