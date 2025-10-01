import os
import glob
import math
import json
import numpy as np
setattr(np, 'complex', complex)
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
DATASET_CSV   = os.path.join(BASE_DIR, 'text', 'XXXX_dataset.csv')
AUDIO_ROOT    = 'XXXX'
OUTPUT_DIR    = os.path.join(BASE_DIR, 'stft_feature')
CHECKPOINT    = os.path.join(OUTPUT_DIR, 'checkpoint.json')
N_FFT         = 1024
HOP_LENGTH    = 256
WIN_LENGTH     = 1024
SPLITS        = 5
# ────────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Read or initialize checkpoint
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'r') as f:
        ckpt = json.load(f)
    last_done = ckpt.get('last_done_split', 0)
else:
    last_done = 0

df = pd.read_csv(DATASET_CSV)
total = len(df)
chunk_size = math.ceil(total / SPLITS)

# 2) Start looping from next_split
for split_idx in range(last_done, SPLITS):
    start_idx = split_idx * chunk_size
    end_idx   = min((split_idx + 1) * chunk_size, total)
    subset    = df.iloc[start_idx:end_idx].reset_index(drop=True)
    if subset.empty:
        last_done = split_idx + 1
        continue

    features = []
    labels   = []

    print(f"\n=== Processing split {split_idx+1}/{SPLITS} "
          f"(rows {start_idx}–{end_idx-1}, {len(subset)} windows) ===")

    for _, row in tqdm(subset.iterrows(),
                       total=len(subset),
                       desc=f"Split {split_idx+1}",
                       unit='win'):
        wav = row['file_name']
        matches = glob.glob(os.path.join(AUDIO_ROOT, '**', wav), recursive=True)
        if not matches:
            continue
        data, sr = sf.read(matches[0], dtype='float32')
        s, e = int(row['start']*sr), int(row['end']*sr)
        seg = data[s:e]
        if seg.size == 0:
            continue

        S = librosa.stft(
            y=seg, n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH
        )
        mag = np.abs(S).T.astype(np.float32)
        features.append(mag)
        labels.append(row['usv'])

    # Save this split
    out_path = os.path.join(OUTPUT_DIR, f'stft_split_{split_idx+1}.npz')
    np.savez_compressed(out_path, features=features, labels=labels)
    print(f"Saved split {split_idx+1} → {out_path}")

    # Update checkpoint
    last_done = split_idx + 1
    with open(CHECKPOINT, 'w') as f:
        json.dump({'last_done_split': last_done}, f)
    print(f"Updated checkpoint: last_done_split = {last_done}")

print("\nAll splits complete.") 
