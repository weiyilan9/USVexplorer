# cli.py

import argparse
import glob
import os

from detector import detect_usv_edges, export_csv
from sync import sync_video
from report import make_report
from config import WAV_PATH, VIDEO_PATH, OUTPUT_CSV, ENABLE_CALIB as CONFIG_ENABLE_CALIB, OUTPUT_MP4, REPORT_PDF

def main():
    parser = argparse.ArgumentParser(description="USVsync command-line interface")
    parser.add_argument(
        '--wav',
        default=WAV_PATH,
        help='Path to WAV file or directory'
    )
    parser.add_argument(
        '--video',
        default=VIDEO_PATH,
        help='Path to MP4 file (set to "none" or leave empty for audio-only)'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Enable calibration mode using labeled snippet'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all WAV files in the specified directory'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        default=True,
        help='Generate PDF report from the CSV output'
    )
    args = parser.parse_args()

    # Determine WAV paths to process
    wav_paths = [args.wav]
    if args.batch and os.path.isdir(args.wav):
        wav_paths = glob.glob(os.path.join(args.wav, '*.wav'))

    for wav in wav_paths:
        # Override calibration flag at runtime
        from config import ENABLE_CALIB
        ENABLE_CALIB = args.calibrate or CONFIG_ENABLE_CALIB

        # Validate WAV file path
        if not wav or not os.path.exists(wav):
            raise FileNotFoundError(f"WAV file not found: {wav}")

        # 1) Detect USV events and export to CSV
        events = detect_usv_edges(wav)
        export_csv(events, OUTPUT_CSV)
        print(f"USV events exported to {OUTPUT_CSV}")

        # 2) Perform audio-video synchronization if a valid video path is provided
        if args.video and args.video.lower() != 'none':
            if not os.path.exists(args.video):
                raise FileNotFoundError(f"Video file not found: {args.video}")
            sync_video(wav, args.video)
            print(f"Synchronized video generated: {os.path.abspath(OUTPUT_MP4)}")

        # 3) Generate PDF report if requested
        if args.report:
            make_report()
            print(f"Report generated: {os.path.abspath(REPORT_PDF)}")

if __name__ == '__main__':
    main()
