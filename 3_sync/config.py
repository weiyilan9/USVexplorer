# config.py
import multiprocessing

# General settings
WAV_PATH        = 'XXXX.wav'
VIDEO_PATH      = 'XXXX.mp4'  # set to None for audio-only mode
OUTPUT_CSV      = 'XXXX.csv'
OUTPUT_SEG_CSV  = 'XXXX.csv'
OUTPUT_MP4      = 'XXXX.mp4'
DEVICE          = 'cuda'  # or 'cpu'
NUM_WORKERS     = multiprocessing.cpu_count()

# Adaptive threshold
ENABLE_ADAPTIVE = True
ADAPTIVE_WIN_S  = 5.0
ADAPTIVE_FACTOR = 1.5

# Automatic threshold
ENABLE_AUTO = True      
AUTO_METHOD = 'otsu' # 'otsu' or 'knee'

# Calibration settings
ENABLE_CALIB    = False
CALIB_AUDIO     = 'XXXX.wav'
CALIB_LABELS    = 'XXXX.csv'
CALIB_GRID_Q    = [0.90, 0.92, 0.94, 0.96, 0.98]

# Reporting
REPORT_PDF      = 'XXXX.pdf'

# Model & window parameters
CKPT_PATH       = 'XXXX.pt'
TARGET_SR       = 250_000
N_FFT           = 1024
HOP             = 256
WIN             = 1024
MICRO_WIN_MS    = 48
MICRO_HOP_MS    = 8
TEMP_SCALE      = 2.0

# Thresholding
HIGH_Q          = 0.95
LOW_FRAC        = 0.48
GAP_ALLOW_MS    = 6
MIN_DUR_MS      = 6
MERGE_IOU       = 0.30

# Sync & visualization
VIS_WIN_SEC     = 0.22
MERGE_NEAR_S    = 1.0
MIN_SEG_SEC     = 1.0
MAX_DISPLAY_SEC = 5.0
FPS_OUT         = 30
FRAME_SIZE      = (1280, 720)
ADD_AUDIO       = True
NFFT_SPEC       = 512
HOP_SPEC        = 128
FMIN            = 20000
FMAX            = 80000

