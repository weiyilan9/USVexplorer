# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, CKPT_PATH, N_FFT

FEATURE_DIM = N_FFT//2 + 1  # 513
D_MODEL     = 768
NUM_HEADS   = 12
NUM_LAYERS  = 8

class BandGate(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(1)
        g = F.relu(self.fc1(g))
        g = torch.sigmoid(self.fc2(g))
        return x * g.unsqueeze(1)

class Conv1dSub(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, D_MODEL, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        return h.transpose(1, 2)

class TransEnc(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, NUM_LAYERS)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

class SegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg  = BandGate()
        self.sub = Conv1dSub()
        self.enc = TransEnc()
        self.cls = nn.Linear(D_MODEL, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bg(x)
        h = self.sub(x)
        h = self.enc(h)
        h = h.mean(1)
        return self.cls(h).squeeze(-1)


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location='cpu', strict=False):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        for k in ['model','state_dict','net']:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def build_model():
    model = SegmentModel().to(DEVICE).eval()
    load_checkpoint(model, CKPT_PATH, map_location=DEVICE, strict=False)
    return model

import numpy as np

def infer_batch(model, segment_array: np.ndarray, temp_scale: float):
    out = []
    with torch.no_grad():
        for i in range(0, len(segment_array), 256):
            batch = torch.from_numpy(segment_array[i:i+256]).float().to(DEVICE)
            logits = model(batch) / temp_scale
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)
    