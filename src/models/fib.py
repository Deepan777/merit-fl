"""Federated IDS Backbone (FIB): CNN-BiLSTM detector (Sec. 3.2).

Tabular flow features are treated as a 1-D sequence: Conv1d extracts spatial
(local feature-group) patterns, BiLSTM captures cross-feature dependencies.
Sized to train comfortably on an RTX 3060 (6GB) and on CPU for smoke tests.
"""
import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int,
                 conv_ch: int = 32, lstm_h: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_ch), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_ch, lstm_h, batch_first=True,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * lstm_h, n_classes))

    def forward(self, x):                  # x: (B, F)
        self.lstm.flatten_parameters()     # deep copies break the cuDNN weight layout
        z = self.conv(x.unsqueeze(1))      # (B, C, F/2)
        z = z.transpose(1, 2)              # (B, F/2, C)
        out, _ = self.lstm(z)
        z = out.mean(dim=1)                # temporal mean-pool
        return self.head(z)

    @torch.no_grad()
    def confidence(self, x):
        """Max softmax prob — gamma_u(t) input to DTRE, and the
        low-confidence gate for the LSRL slow path."""
        p = torch.softmax(self.forward(x), dim=1)
        conf, pred = p.max(dim=1)
        return conf, pred


class MLPBaseline(nn.Module):
    """Local-only / centralized baseline detector."""
    def __init__(self, n_features, n_classes, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, n_classes))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def count_flops(model, n_features):
    """Forward FLOPs per sample (multiply-add = 2 FLOPs) of Conv1d, Linear
    and LSTM layers. Element-wise ops, BatchNorm and pooling are excluded."""
    total = [0]

    def conv(m, inp, out):
        total[0] += 2 * out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin(m, inp, out):
        total[0] += 2 * out.numel() * m.in_features

    def lstm(m, inp, out):
        steps = inp[0].shape[1] if m.batch_first else inp[0].shape[0]
        dirs, h, x = (2 if m.bidirectional else 1), m.hidden_size, m.input_size
        for _ in range(m.num_layers):
            total[0] += dirs * steps * 2 * 4 * h * (x + h)
            x = dirs * h

    hooks = [mod.register_forward_hook(fn) for mod in model.modules()
             for typ, fn in ((nn.Conv1d, conv), (nn.Linear, lin), (nn.LSTM, lstm))
             if isinstance(mod, typ)]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, n_features, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(total[0])


def build_model(name, n_features, n_classes):
    if name == "cnn_bilstm":
        return CNNBiLSTM(n_features, n_classes)
    if name == "mlp":
        return MLPBaseline(n_features, n_classes)
    raise ValueError(name)
