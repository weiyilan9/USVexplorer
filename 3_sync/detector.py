# detector.py

import numpy as np
import librosa
import pandas as pd
from config import *
from model import build_model, infer_batch
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing


def compute_segments(y, sr):
    # Compute full STFT once on GPU-capable device if needed
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mag = np.abs(S).T.astype(np.float32)
    n_frames = mag.shape[0]
    frame_time = HOP / sr
    win = int(round((MICRO_WIN_MS / 1000) * sr / HOP))
    hop_frames = int(round((MICRO_HOP_MS / 1000) * sr / HOP))
    starts = np.arange(0, n_frames - win + 1, hop_frames)
    segments = mag[starts[:, None] + np.arange(win)[None, :]]
    times = starts * frame_time + (win * frame_time / 2)
    return segments, times


def time_to_seconds(time_str):
    mins, sec_frac = time_str.split(':')
    secs, frac = sec_frac.split('.')
    return int(mins) * 60 + int(secs) + float(f"0.{frac}")


def evaluate_against_labels(times, probs, thr, low_thr, labels_df):
    labels_df.columns = labels_df.columns.str.strip()
    preds = []
    mask = probs >= low_thr
    # morphological closing to fill small gaps
    mask = binary_closing(mask, structure=np.ones(3))
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            # require at least one point above thr
            if np.any(probs[i:j] >= thr):
                preds.append((times[i], times[j-1]))
            i = j
        else:
            i += 1

    tp = 0
    for pt, pe in preds:
        for _, row in labels_df.iterrows():
            ls = time_to_seconds(row['Start'])
            ld = time_to_seconds(row['Duration'])
            le = ls + ld
            inter = max(0, min(pe, le) - max(pt, ls))
            union = (pe - pt) + (le - ls) - inter
            if union > 0 and inter / union > 0.5:
                tp += 1
                break

    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(labels_df) if len(labels_df) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def calibrate_thresholds(model, segments, times):
    labels = pd.read_csv(CALIB_LABELS, sep='\t')
    labels.columns = labels.columns.str.strip()
    best = (HIGH_Q, LOW_FRAC, 0.0)
    probs = infer_batch(model, segments, TEMP_SCALE)
    for q in CALIB_GRID_Q:
        thr_q = np.quantile(probs, q)
        low_q = thr_q * LOW_FRAC
        f1 = evaluate_against_labels(times, probs, thr_q, low_q, labels)
        if f1 > best[2]:
            best = (q, low_q / thr_q, f1)
    return best[0], best[1]


def adaptive_thresholds(probs, times):
    win = int(ADAPTIVE_WIN_S / (HOP / TARGET_SR))
    low_thr = []
    for i in range(len(probs)):
        lo = np.quantile(probs[max(0, i - win):i + win + 1], 0.1)
        low_thr.append(lo * ADAPTIVE_FACTOR)
    return np.array(low_thr)


def auto_threshold_knee(probs, low_frac, qs=np.linspace(0.6, 0.98, 20)):
    counts = []
    for q in qs:
        thr = np.quantile(probs, q)
        mask = probs >= thr * low_frac
        edges = np.diff(mask.astype(int)) != 0
        n_events = int(np.sum(mask[0]) + np.sum(edges) / 2)
        counts.append(n_events)
    diffs2 = np.diff(counts, n=2)
    idx = np.argmax(-diffs2) + 1
    return qs[min(idx, len(qs)-1)]


def merge_iou(events, iou_thresh):
    events = sorted(events, key=lambda x: x['score'], reverse=True)
    merged = []
    for ev in events:
        if not any(
            (max(0, min(ev['end'], m['end']) - max(ev['start'], m['start'])) /
             ((ev['end'] - ev['start']) + (m['end'] - m['start']) -
              max(0, min(ev['end'], m['end']) - max(ev['start'], m['start']))))
            >= iou_thresh for m in merged
        ):
            merged.append(ev)
    return merged


def detect_usv_edges(wav_path=WAV_PATH):
    # load audio
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # compute segments and times once
    segments, times = compute_segments(y, sr)
    model = build_model()

    # calibration if enabled
    hq, lf = HIGH_Q, LOW_FRAC
    if ENABLE_CALIB:
        hq, lf = calibrate_thresholds(model, segments, times)

    # inference: single pass
    probs = infer_batch(model, segments, TEMP_SCALE)

    # threshold selection
    if ENABLE_AUTO and not ENABLE_CALIB:
        thr_quant = np.quantile(probs, HIGH_Q)
        if AUTO_METHOD.lower() == 'otsu':
            thr_auto = threshold_otsu(probs)
        else:
            best_q = auto_threshold_knee(probs, lf)
            thr_auto = np.quantile(probs, best_q)
        thr = max(thr_auto, thr_quant)
        low_thr = thr * lf
    elif ENABLE_ADAPTIVE and not ENABLE_CALIB:
        low_thr = adaptive_thresholds(probs, times)
        thr = None
    else:
        thr = np.quantile(probs, hq)
        low_thr = np.full_like(probs, thr * lf)

    # event extraction with stricter mask
    events = []
    mask = probs >= low_thr
    half_win = (MICRO_WIN_MS / 1000) / 2
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            # ensure at least one point meets high threshold
            if thr is None or np.any(probs[i:j] >= thr):
                start = times[i] - half_win
                end = times[j-1] + half_win
                events.append({
                    'start': max(0, start),
                    'end':   end,
                    'score': float(probs[i:j].max())
                })
            i = j
        else:
            i += 1

    # merge overlaps
    events = merge_iou(events, MERGE_IOU)
    return sorted(events, key=lambda x: x['start'])


def export_csv(events, csv_path=OUTPUT_CSV):
    """
    Export detailed USV event metrics to CSV, including:
      - event_id
      - start_time_s, end_time_s, duration_ms
      - peak_score
      - peak/min/max/mean/median frequency (kHz)
      - bandwidth (kHz)
      - spectral centroid & bandwidth (kHz)
      - FM slope (kHz/s)
      - RMS amplitude
      - spectral flatness
      - inter-call interval (ms)
    """
    # load full audio once
    y, sr = librosa.load(WAV_PATH, sr=TARGET_SR, mono=True)
    rows = []
    prev_start = None

    for idx, ev in enumerate(events):
        start, end, score = ev['start'], ev['end'], ev['score']
        dur = end - start
        # extract segment
        i0, i1 = int(start * sr), int(end * sr)
        y_seg = y[i0:i1]

        # STFT for spectral features
        D = librosa.stft(y_seg, n_fft=NFFT_SPEC, hop_length=HOP_SPEC)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT_SPEC)

        # per-frame peak freq (Hz)
        peak_bin_idxs = np.argmax(S, axis=0)
        peak_freqs = freqs[peak_bin_idxs]

        # basic freq stats (in kHz)
        min_f = float(np.min(peak_freqs)) / 1000.0
        max_f = float(np.max(peak_freqs)) / 1000.0
        mean_f = float(np.mean(peak_freqs)) / 1000.0
        median_f = float(np.median(peak_freqs)) / 1000.0
        bandwidth = max_f - min_f
        peak_freq = max_f

        # spectral centroid & bandwidth (Hz → kHz)
        cent = librosa.feature.spectral_centroid(
            y=y_seg, sr=sr, n_fft=NFFT_SPEC, hop_length=HOP_SPEC
        )[0].mean() / 1000.0
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y_seg, sr=sr, n_fft=NFFT_SPEC, hop_length=HOP_SPEC
        )[0].mean() / 1000.0

        # FM slope (kHz/s)
        fm_slope = bandwidth / dur if dur > 0 else np.nan

        # RMS amplitude
        rms = librosa.feature.rms(y=y_seg)[0].mean()

        # spectral flatness
        flat = librosa.feature.spectral_flatness(
            y=y_seg, n_fft=NFFT_SPEC, hop_length=HOP_SPEC
        )[0].mean()

        # inter-call interval
        if prev_start is None:
            interval = np.nan
        else:
            interval = (start - prev_start) * 1000.0
        prev_start = start

        rows.append({
            'event_id':             f'USV_{idx+1:03d}',
            'start_time_s':         start,
            'end_time_s':           end,
            'duration_ms':          dur * 1000.0,
            'peak_score':           score,
            'peak_freq_khz':        peak_freq,
            'min_freq_khz':         min_f,
            'max_freq_khz':         max_f,
            'mean_freq_khz':        mean_f,
            'median_freq_khz':      median_f,
            'bandwidth_khz':        bandwidth,
            'centroid_khz':         cent,
            'spectral_bandwidth_khz': spec_bw,
            'fm_slope_khz_per_s':   fm_slope,
            'rms_amplitude':        rms,
            'spectral_flatness':    flat,
            'call_interval_ms':     interval,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df
