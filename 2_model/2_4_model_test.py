import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────
FEAT_DIR   = "XXXX/stft_feature"
OUT_DIR    = "XXXX/output"
CKPT_PATH  = os.path.join(OUT_DIR, "XXXX.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE, NUM_WORKERS = 64, 16
FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS = 513, 768, 12, 8

TEST_FEAT = os.path.join(FEAT_DIR, "XXXX_test_features.npy")
TEST_LAB  = os.path.join(FEAT_DIR, "XXXX_test_labels.npy")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────
class SegmentDataset(Dataset):
    def __init__(self, feat_file, lab_file):
        X = np.load(feat_file, allow_pickle=True)
        Y = np.load(lab_file,  allow_pickle=True)
        self.X = [torch.from_numpy(x.astype(np.float32)) for x in X]
        self.Y = torch.from_numpy(Y.astype(np.int64))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

test_loader = DataLoader(
    SegmentDataset(TEST_FEAT, TEST_LAB),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ─── Model  ────────────────────────────────────
class BandGate(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim//r, bias=False)
        self.fc2 = nn.Linear(dim//r, dim, bias=False)
    def forward(self, x):
        # x: (B, T, D)
        g = torch.sigmoid(self.fc2(
            nn.functional.relu(self.fc1(x.mean(1)))
        ))
        return x * g.unsqueeze(1)

class Conv1dSub(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, D_MODEL, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(D_MODEL,      D_MODEL, 3, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        h = self.net(x.transpose(1,2))
        return h.transpose(1,2)  # back to (B, T/4, D_MODEL)

class TransEnc(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
    def forward(self, x):
        return self.enc(x)

class SegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg  = BandGate(FEATURE_DIM)
        self.sub = Conv1dSub()
        self.enc = TransEnc()
        self.cls = nn.Linear(D_MODEL, 1)
    def forward(self, x):
        # x: (B, T, D)
        x = self.bg(x)            
        h = self.sub(x)          
        h = self.enc(h)          
        return self.cls(h.mean(1)).squeeze(-1)  # (B,)

model = SegmentModel().to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

# ─── Inference ───────────────────────────────────────────────
probs, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        trues.extend(y.numpy().tolist())

# ─── Dynamic Thresholds & Indicators ─────────────────────────────────────────
prec_vals, rec_vals, thr = precision_recall_curve(trues, probs)
f1s = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-8)
idx = np.argmax(f1s)
best_thr = thr[idx] if idx < len(thr) else 1.0
preds = (np.array(probs) > best_thr).astype(int)

precision = precision_score(trues, preds)
recall    = recall_score(trues, preds)
binary_f1 = f1_score(trues, preds)
mcc       = matthews_corrcoef(trues, preds)
pr_auc    = average_precision_score(trues, probs)
roc_auc   = roc_auc_score(trues, probs)

# ─── Print & Save Indicators ─────────────────────────────────────────
print("\n=== Test Results ===")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1: {binary_f1:.4f}")
print(f"MCC              : {mcc:.4f}")
print(f"PR AUC           : {pr_auc:.4f}")
print(f"ROC AUC          : {roc_auc:.4f}")

metrics_path = os.path.join(OUT_DIR, "eval_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Precision       : {precision:.4f}\n")
    f.write(f"Recall          : {recall:.4f}\n")
    f.write(f"Segment Binary F1: {binary_f1:.4f}\n")
    f.write(f"MCC              : {mcc:.4f}\n")
    f.write(f"PR AUC           : {pr_auc:.4f}\n")
    f.write(f"ROC AUC          : {roc_auc:.4f}\n")

# ─── Visualization: PR Curve ─────────────────────────────────────────
plt.figure(figsize=(6,6))
plt.plot(rec_vals, prec_vals, label=f"PR AUC={pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "eval_pr_curve.png"), dpi=300)
plt.close()

# ─── Visualization: ROC Curve ───────────────────────────────────────
fpr, tpr, _ = roc_curve(trues, probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "eval_roc_curve.png"), dpi=300)
plt.close()

# ─── Visualization: PCA Scatter Plot ─────────────────────────────────────
segments = np.load(TEST_FEAT, allow_pickle=True)
labels   = np.load(TEST_LAB,  allow_pickle=True).astype(int)
X = np.array([seg.mean(axis=0) for seg in segments])
Z = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6,6))
plt.scatter(Z[:,0], Z[:,1], c=labels, cmap='viridis', alpha=0.7)
plt.title("PCA of Test Segments")
plt.savefig(os.path.join(OUT_DIR, "eval_pca.png"), dpi=300)
plt.close()

print("\nAll evaluation artifacts saved to", OUT_DIR)
