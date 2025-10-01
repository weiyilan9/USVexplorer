# readme

This repository contains the supporting materials for **USVexplorer: Robust Detection of  Ultrasonic Vocalizations with Cross Species Generalization**, a framework designed to detect ultrasonic vocalizations (USVs) across species and synchronize them with corresponding behavioral video recordings. The materials are organized into three main modules:

```bash
├── 1_dataset_construction
│   ├── 1_1_RatPup_dataset.ipynb
│   ├── 1_2_DeepSqueak_dataset.ipynb
│   ├── 1_3_NABat_dataset.ipynb
│   ├── 1_4_MarmAudio_dataset.ipynb
├── 2_model
│   ├── 2_1_feature_extract.py
│   ├── 2_2_merge_feature.py
│   ├── 2_3_model_train.py
│   └── 2_4_model_test.py
├── 3_sync
│   ├── cli.py
│   ├── config.py
│   ├── detector.py
│   ├── model.py
│   ├── report.py
│   └── sync.py
└── readme.md

```

## Module Overview

### 1. 1_dataset_construction

This folder includes Jupyter notebooks for preparing four datasets:

- `1_1_RatPup_dataset.ipynb`: Preprocessing RatPup rat recordings
- `1_2_DeepSqueak_dataset.ipynb`: Preprocessing DeepSqueak rat recordings
- `1_3_NABat_dataset.ipynb`: Preprocessing NABat bat recordings
- `1_4_MarmAudio_dataset.ipynb`: Processing MarmAudio marmoset recordings

### 2. 2_model

Python scripts for acoustic feature processing, model training, and evaluation:

| File | Description |
| --- | --- |
| `2_1_feature_extract.py` | Extracts basic acoustic features from labeled segments |
| `2_2_merge_feature.py` | Combines features from multiple datasets into a unified matrix |
| `2_3_model_train.py` | Trains a classification model using extracted features |
| `2_4_model_test.py` | Evaluates and visualizes model performance on test data |

### 3. 3_sync

This module aligns audio and video recordings and generates multimodal visualizations:

| File | Description |
| --- | --- |
| `cli.py` | Entry point for running the entire sync pipeline from the command line |
| `config.py` | Contains key parameters for audio sampling, frame rates, and plotting etc. |
| `detector.py` | Lightweight USV detector based on frequency heuristics |
| `model.py` | Load pre-trained model for event prediction |
| `sync.py` | Performs time offset correction and generates synchronized segments |
| `report.py` | Produces synchronized audio-video visualizations and PDF reports |

## Requirements

Below is a list of required Python packages to run the scripts:

```bash
numpy
scipy
pandas
matplotlib
scikit-learn
librosa
tqdm
Pillow
imageio
ffmpeg-python
```

## Quick Start

Example: Running the synchronization pipeline from the `3_sync` folder.

### 1. Prepare audio and video files

Ensure audio and video files have matching timestamps in their filenames, such as:

```bash
example_2025-09-01_10-00-00.wav
example_2025-09-01_10-00-00.mp4
```

### 2. Run the main script

```bash
cd 3_sync
python cli.py --audio path/to/audio.wav --video path/to/video.mp4
```

### 3. Output files

- `usv_events.csv`: Detected USV segments
- `sync_output.mp4`: Synchronized multimodal video
- `sync_report.pdf`: PDF report with visualizations
