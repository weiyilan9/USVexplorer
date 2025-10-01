import os, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ────── Configuration ──────
FEAT_DIR   = "XXXX/stft_feature"
OUT_DIR    = "XXXX/output"; os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS = 513, 768, 12, 8
BATCH_SIZE, NUM_WORKERS = 64, 16
LR, WD = 1e-4, 1e-2
TOTAL_STEPS = 10_000; WARMUP_STEPS = int(0.2*TOTAL_STEPS); VALID_EVERY = 100
CLIP_NORM = 5.0

# ────── Dataset & Sampling ──────
class SegmentDataset(Dataset):
    def __init__(self, feat, lab):
        X = np.load(feat, allow_pickle=True); Y = np.load(lab, allow_pickle=True)
        self.X = [torch.from_numpy(x.astype(np.float32)) for x in X]
        self.Y = torch.from_numpy(Y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

class BalancedSampler(Sampler):
    def __init__(self, pos, neg, b):
        self.pos, self.neg, self.b, self.k = pos, neg, b, b//2
    def __iter__(self):
        p, n = np.random.permutation(self.pos), np.random.permutation(self.neg)
        ip = in_ = 0
        while ip < len(p):
            if in_ + self.b - self.k > len(n):
                n, in_ = np.random.permutation(self.neg), 0
            batch = np.concatenate([p[ip:ip+self.k], n[in_:in_+self.b-self.k]])
            np.random.shuffle(batch)
            for i in batch: yield int(i)
            ip += self.k; in_ += self.b-self.k
    def __len__(self): return max(1,(len(self.pos)//self.k)*self.b)

def loader(split, shuffle=False, sampler=None):
    f = f"{FEAT_DIR}/XXXX_{split}_features.npy"
    l = f"{FEAT_DIR}/XXXX_{split}_labels.npy"
    return DataLoader(SegmentDataset(f,l), BATCH_SIZE,
                      shuffle=shuffle, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True)

# ────── Model ──────
class BandGate(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim//r, bias=False)
        self.fc2 = nn.Linear(dim//r, dim, bias=False)
    def forward(self, x):                       # x:(B,T,D)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(1)))))  # (B,D)
        return x * g.unsqueeze(1)

class Conv1dSub(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, D_MODEL, 3, 2, 1), nn.ReLU(),
            nn.Conv1d(D_MODEL,      D_MODEL, 3, 2, 1), nn.ReLU())
    def forward(self,x): return self.net(x.transpose(1,2)).transpose(1,2)

class TransEnc(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D_MODEL, NUM_HEADS, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, NUM_LAYERS)
    def forward(self,x): return self.enc(x)

class SegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg  = BandGate(FEATURE_DIM)
        self.sub = Conv1dSub(); self.enc = TransEnc(); self.cls = nn.Linear(D_MODEL,1)
    def forward(self,x):
        x = self.bg(x); h = self.sub(x); h = self.enc(h)
        return self.cls(h.mean(1)).squeeze(-1)

# ────── Data loading ──────
train_ds = SegmentDataset(f"{FEAT_DIR}/XXXX_train_features.npy",
                          f"{FEAT_DIR}/XXXX_train_labels.npy")
pos = [i for i,y in enumerate(train_ds.Y) if y==1]; neg=[i for i in range(len(train_ds)) if i not in pos]
train_loader = loader("train", sampler=BalancedSampler(pos,neg,BATCH_SIZE))
val_loader   = loader("val")
test_loader  = loader("test")

# ────── Training preparation ──────
model = SegmentModel().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = SequentialLR(
    opt,
    [
        LinearLR(opt, start_factor=0.1, total_iters=WARMUP_STEPS),
        CosineAnnealingLR(opt, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=1e-6)
    ],
    milestones=[WARMUP_STEPS]
)

pos_weight = torch.tensor([len(neg)/len(pos)], device=DEVICE)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scaler, step, best_f1 = torch.cuda.amp.GradScaler(), 0, 0.0

# ────── Training loop ──────
while step < TOTAL_STEPS:
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1

        if step % 50 == 0: 
            print(f"Step {step:<6d} | lr={opt.param_groups[0]['lr']:.2e} | loss={loss.item():.4f}")

        if step % VALID_EVERY == 0:
            model.eval(); prob, true = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    p = torch.sigmoid(model(vx.to(DEVICE))).cpu().numpy()
                    prob.extend(p.tolist()); true.extend(vy.numpy().tolist())

            prec, rec, thr = precision_recall_curve(true, prob)
            f1s = 2 * prec * rec / (prec + rec + 1e-8)
            idx = np.argmax(f1s)
            best_thr = thr[idx] if idx < len(thr) else 1.0
            val_f1   = f1s[idx]

            print(f"── Val @ step {step}: F1={val_f1:.4f} | thr={best_thr:.3f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(),
                           f"{OUT_DIR}/best_step{step}_f1{val_f1:.3f}.pt")
            model.train()

        if step >= TOTAL_STEPS:
            break

# ────── Test Evaluation ──────
model.eval(); prob, true = [], []
with torch.no_grad():
    for x, y in test_loader:
        p = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        prob.extend(p.tolist()); true.extend(y.numpy().tolist())

prec, rec, thr = precision_recall_curve(true, prob)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
idx = np.argmax(f1s)
best_thr = thr[idx] if idx < len(thr) else 1.0

y_pred = (np.array(prob) > best_thr).astype(int)

metrics = dict(
    f1      = f1_score(true, y_pred),
    mcc     = matthews_corrcoef(true, y_pred),
    pr_auc  = average_precision_score(true, prob),
    roc_auc = roc_auc_score(true, prob),
    thr     = best_thr
)
with open(f"{OUT_DIR}/test_metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k.upper():7s}: {v:.4f}\n")
print(open(f"{OUT_DIR}/test_metrics.txt").read())

# ────── Curves & PCA ──────
plt.figure(); plt.plot(rec,prec,label=f"PR AUC={metrics['pr_auc']:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(); plt.legend()
plt.title("PR Curve"); plt.savefig(f"{OUT_DIR}/pr_curve.png",dpi=300); plt.close()

fpr,tpr,_ = roc_curve(true, prob)
plt.figure(); plt.plot(fpr,tpr,label=f"ROC AUC={metrics['roc_auc']:.3f}")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(); plt.legend()
plt.title("ROC Curve"); plt.savefig(f"{OUT_DIR}/roc_curve.png",dpi=300); plt.close()

feats = np.load(f"{FEAT_DIR}/XXXX_test_features.npy",allow_pickle=True)
labs  = np.load(f"{FEAT_DIR}/XXXX_test_labels.npy",allow_pickle=True)
Z = PCA(2).fit_transform(np.array([seg.mean(0) for seg in feats]))
plt.figure(figsize=(6,6))
plt.scatter(Z[:,0],Z[:,1],c=labs.astype(int),cmap='viridis',alpha=.7)
plt.title("PCA of Test Segments"); plt.savefig(f"{OUT_DIR}/pca.png",dpi=300); plt.close()
