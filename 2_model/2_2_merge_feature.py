import os, math
import numpy as np
import pandas as pd
from glob import glob

# ── Configuration ─────────────────────────────────────────────────────────────────────
npz_dir  = "XXXX"
csv_path = "XXXX_dataset.csv"
out_dir  = npz_dir
os.makedirs(out_dir, exist_ok=True)

# ── 1) Read CSV ────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)

# ── 2) Find all split npz and count the number of rows corresponding to each block ─────
npz_list   = sorted(glob(os.path.join(npz_dir, "stft_split_*.npz")))
num_splits = len(npz_list)
total_rows = len(df)
chunk_size = math.ceil(total_rows / num_splits)

# ── 3) Initialize the container ────────────────────────────────────────────────────
splits = {
    "train": ([], []),
    "val":   ([], []),
    "test":  ([], []),
}

# ── 4) Traverse each split and assign it according to the CSV type ─────────────────
for idx, npz_file in enumerate(npz_list):
    start = idx * chunk_size
    end   = min((idx + 1) * chunk_size, total_rows)
    sub_df = df.iloc[start:end].reset_index(drop=True)

    arr    = np.load(npz_file, allow_pickle=True)
    feats  = arr["features"]
    labs   = arr["labels"]

    if len(feats) != len(sub_df):
        raise RuntimeError(
            f"The length of the {idx+1}th block is inconsistent: npz has {len(feats)} rows, CSV has {len(sub_df)} rows"
        )

    for i, feat in enumerate(feats):
        tp = sub_df.loc[i, "type"]  # train/val/test
        splits[tp][0].append(feat)
        splits[tp][1].append(labs[i])

# ── 5) Save as .npy ─────────────────────────────────────────────────────────────
for tp, (feat_list, lab_list) in splits.items():
    feats_arr = np.array(feat_list, dtype=object)
    labs_arr  = np.array(lab_list,  dtype=object)
    np.save(os.path.join(out_dir, f"RatPup_{tp}_features.npy"), feats_arr)
    np.save(os.path.join(out_dir, f"RatPup_{tp}_labels.npy"),   labs_arr)

print("All done! Output directory: ", out_dir)
