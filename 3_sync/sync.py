# sync.py

import os
import subprocess
import re
import imageio.v2 as imageio
import pandas as pd
import librosa
import shutil
from tqdm import tqdm
from detector import detect_usv_edges, export_csv
from config import (
    WAV_PATH, VIDEO_PATH,
    OUTPUT_CSV, OUTPUT_SEG_CSV, OUTPUT_MP4,
    TARGET_SR, ADD_AUDIO,
    MERGE_NEAR_S, MIN_SEG_SEC,
    FPS_OUT, FRAME_SIZE,
    MAX_DISPLAY_SEC,
    NFFT_SPEC, HOP_SPEC, FMIN, FMAX
)
from PIL import Image
import numpy as np
from datetime import datetime


def calculate_time_offset_by_filename(audio_path, video_path):
    def _parse(name):
        patterns = [
            (r"(20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", "%Y-%m-%d_%H-%M-%S"),
            (r"(20\d{6}_\d{6})", "%Y%m%d_%H%M%S"),
        ]
        base = os.path.basename(name)
        for pat, fmt in patterns:
            m = re.search(pat, base)
            if m:
                try:
                    return datetime.strptime(m.group(1), fmt)
                except:
                    pass
        return None

    ta = _parse(audio_path)
    tv = _parse(video_path)
    if ta and tv:
        return (ta - tv).total_seconds()
    else:
        return 0.0


def merge_nearby(events):
    if not events:
        return []
    segs = []
    cur = {'start': events[0]['start'], 'end': events[0]['end'], 'usvs': [events[0]]}
    for ev in events[1:]:
        s, e = ev['start'], ev['end']
        if s <= cur['end'] + MERGE_NEAR_S:
            cur['end'] = max(cur['end'], e)
            cur['usvs'].append(ev)
        else:
            segs.append(cur)
            cur = {'start': s, 'end': e, 'usvs': [ev]}
    segs.append(cur)
    # enforce minimum duration
    for seg in segs:
        dur = seg['end'] - seg['start']
        if dur < MIN_SEG_SEC:
            ext = (MIN_SEG_SEC - dur) / 2
            seg['start'] = max(0.0, seg['start'] - ext)
            seg['end'] = seg['start'] + MIN_SEG_SEC
    return segs


def create_segment_spectrogram(y, sr, start, duration):
    s = int(start * sr)
    e = min(int((start + duration) * sr), len(y))
    seg = y[s:e]
    D = librosa.stft(seg, n_fft=NFFT_SPEC, hop_length=HOP_SPEC)
    S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT_SPEC)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_SPEC) + start
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    return {
        'frequencies': freqs[mask],
        'times': times,
        'spectrogram_db': S_db[mask]
    }


def create_combined_frame(spec_data, segment, frame, current_time, frame_size=FRAME_SIZE):
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(frame_size[0] / 100, frame_size[1] / 100),
        dpi=100
    )
    im = ax1.imshow(
        spec_data['spectrogram_db'],
        aspect='auto',
        origin='lower',
        extent=[
            spec_data['times'][0], spec_data['times'][-1],
            spec_data['frequencies'][0] / 1000, spec_data['frequencies'][-1] / 1000
        ],
        cmap='viridis'
    )
    ax1.axvline(current_time, color='white', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Freq (kHz)')
    plt.colorbar(im, ax=ax1)

    if frame is not None:
        ax2.imshow(frame)
    else:
        ax2.text(0.5, 0.5, 'No Video', ha='center', va='center')
    ax2.axis('off')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    buf.close()
    plt.close(fig)

    if img.size != frame_size:
        img = img.resize(frame_size, Image.Resampling.LANCZOS)
    return np.array(img)


def extract_audio_segment(y, sr, start, duration, out_wav):
    import soundfile as sf
    s = int(start * sr)
    e = min(int((start + duration) * sr), len(y))
    sf.write(out_wav, y[s:e], sr)


def sync_video(wav=WAV_PATH, video=VIDEO_PATH):
    # 1. detect and export events
    events = detect_usv_edges(wav)
    df_events = export_csv(events, OUTPUT_CSV)

    # 2. if no video, return
    if not video:
        return df_events

    # 3. merge and split long segments
    raw_segs = merge_nearby(events)
    split_segs = []
    for seg in raw_segs:
        start, end = seg['start'], seg['end']
        usvs = seg['usvs']
        if end - start <= MAX_DISPLAY_SEC:
            split_segs.append(seg)
        else:
            cur = start
            while cur < end:
                nxt = min(cur + MAX_DISPLAY_SEC, end)
                window_usvs = [u for u in usvs if u['start'] >= cur and u['start'] < nxt]
                if window_usvs:
                    split_segs.append({'start': cur, 'end': nxt, 'usvs': window_usvs})
                cur = nxt
    segs = split_segs

    # export segments CSV
    df_segs = pd.DataFrame([{
        'segment_start_s': s['start'],
        'segment_end_s':   s['end'],
        'n_usvs':          len(s['usvs'])
    } for s in segs])
    df_segs.to_csv(OUTPUT_SEG_CSV, index=False)

    # 4. load audio
    y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)

    # 5. compute offset
    offset = calculate_time_offset_by_filename(wav, video)

    # 6. open video
    reader = imageio.get_reader(video, format='ffmpeg')
    try:
        fps = reader.get_meta_data().get('fps', FPS_OUT)
    except:
        fps = FPS_OUT

    # 7. prepare writer
    tmp_dir = '_tmp_usv'
    os.makedirs(tmp_dir, exist_ok=True)
    writer = imageio.get_writer(
        os.path.join(tmp_dir, 'video_no_audio.mp4'),
        format='ffmpeg',
        fps=FPS_OUT,
        codec='libx264',
        output_params=['-pix_fmt', 'yuv420p', '-crf', '23']
    )

    # 8. render each segment with progress bars
    print("Start rendering segments...")
    for seg_idx, seg in enumerate(tqdm(segs, desc="Segments", unit="seg")):
        start = seg['start']
        dur = seg['end'] - start
        spec_data = create_segment_spectrogram(y, sr, start, dur)

        # export audio segment
        seg_wav = os.path.join(tmp_dir, f'seg_{int(start*1000)}.wav')
        extract_audio_segment(y, sr, start, dur, seg_wav)

        n_frames = int(np.ceil(dur * FPS_OUT))
        for _ in tqdm(range(n_frames),
                      desc=f"Seg {seg_idx+1}/{len(segs)}",
                      leave=False,
                      unit="frame"):
            t_audio = start + (_ / FPS_OUT)
            t_video = t_audio - offset
            frame = None
            try:
                reader.set_image_index(int(t_video * fps))
                frame = reader.get_next_data()
            except:
                pass
            img = create_combined_frame(spec_data, seg, frame, t_audio)
            writer.append_data(img)

    reader.close()
    writer.close()

    # 9. mux audio
    if ADD_AUDIO:
        print("Muxing audio…")
        lst_path = os.path.join(tmp_dir, 'list.txt')
        with open(lst_path, 'w') as f:
            for s in segs:
                wav_name = f"seg_{int(s['start']*1000)}.wav"
                abs_path = os.path.abspath(os.path.join(tmp_dir, wav_name))
                f.write(f"file '{abs_path}'\n")

        combined_audio = os.path.join(tmp_dir, 'combined_audio.wav')
        subprocess.run([
            'ffmpeg','-hide_banner','-loglevel','error',
            '-f','concat','-safe','0','-i',lst_path,
            '-c','copy','-y',combined_audio
        ], check=True)
        subprocess.run([
            'ffmpeg','-hide_banner','-loglevel','error',
            '-i', os.path.join(tmp_dir,'video_no_audio.mp4'),
            '-i', combined_audio,
            '-c:v','copy','-c:a','aac','-y', OUTPUT_MP4
        ], check=True)

    # 10. cleanup temp dir
    print("Cleaning up temporary files…")
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"Warning: failed to remove temp dir {tmp_dir}: {e}")

    print("Done syncing video.")
    return df_events, df_segs
