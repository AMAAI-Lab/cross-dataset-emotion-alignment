import torch.nn as nn
from torch.nn import LayerNorm, TransformerEncoderLayer


class MusicMoodModel(nn.Module):
    def __init__(self, d_model, output_size, num_layers=3, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [3, 6, 9, 12]

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(
                        self.d_model * len(self.selected_layers),
                        self.d_model * len(self.selected_layers),
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(self.d_model * len(self.selected_layers), output_size))

        self.sequential = nn.Sequential(*layers)

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)
        return self.sequential(x)


class MusicMoodModelLayer6(nn.Module):
    def __init__(self, d_model, output_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [6]

        self.sequential = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model * 4, 4 * output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * output_size, output_size),
        )

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)
        return self.sequential(x)


class MusicMoodModelWithAttention(nn.Module):
    def __init__(self, d_model, output_size, nhead=8, num_layers=3, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [3, 6, 9, 12]

        # Input projection
        self.input_proj = nn.Linear(4 * self.d_model, self.d_model)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=4 * self.d_model,
                    dropout=dropout_rate,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norm = LayerNorm(self.d_model)

        # Output projection
        self.output_proj = nn.Linear(self.d_model, output_size)

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)

        # Project input to d_model dimensions
        x = self.input_proj(x)

        # Reshape for self-attention (seq_len=1 as we're dealing with a single "token" per sample)
        x = x.unsqueeze(0)  # Shape: (1, batch_size, d_model)

        # Apply self-attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Reshape and project to output size
        x = x.squeeze(0)  # Shape: (batch_size, d_model)
        output = self.output_proj(x)

        return output
