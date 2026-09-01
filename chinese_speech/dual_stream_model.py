from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class DualStreamGRUDecoder(nn.Module):
    def __init__(
        self,
        *,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_syllable_classes: int,
        n_tone_classes: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 5,
        patch_size: int = 0,
        patch_stride: int = 0,
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_syllable_classes = n_syllable_classes
        self.n_tone_classes = n_tone_classes
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        input_size = self.neural_dim
        if self.patch_size > 0:
            input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=rnn_dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.syllable_head = nn.Linear(self.n_units, self.n_syllable_classes)
        self.tone_head = nn.Linear(self.n_units, self.n_tone_classes)
        nn.init.xavier_uniform_(self.syllable_head.weight)
        nn.init.xavier_uniform_(self.tone_head.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Dict[str, torch.Tensor]:
        day_weights = torch.stack([self.day_weights[int(i)] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[int(i)] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)

        if self.patch_size > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x = x.unfold(3, self.patch_size, self.patch_stride)
            x = x.squeeze(2).permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), x.size(1), -1)

        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        output, hidden_states = self.gru(x, states)
        result = {
            "syllable_logits": self.syllable_head(output),
            "tone_logits": self.tone_head(output),
        }
        if return_state:
            result["hidden_states"] = hidden_states
        return result
