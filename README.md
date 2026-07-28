# ASL Translator

A real-time American Sign Language (ASL) word and phrase translator that runs entirely in the browser. Hand gestures are captured via webcam, tracked using MediaPipe Hands, and classified by an LSTM sequence model converted to TensorFlow.js for fully client-side inference.

## Overview

This project recognizes a fixed vocabulary of 40 common ASL words from live webcam video, using hand landmark tracking rather than raw video classification. No video data ever leaves the user's browser — all inference runs client-side.

## Vocabulary (v1 — 40 words)

hello, thank you, please, sorry, yes, no, help, want, need, like, love, more, finish, again, learn, understand, know, name, my, you, friend, family, mom, dad, home, school, work, eat, drink, water, hungry, happy, sad, sick, tired, good, bad, where, what, who

## Architecture

1. **Data source** — Video clips for each vocabulary word are sourced from the WLASL (Word-Level American Sign Language) dataset, filtered down to only this project's 40-word vocabulary via a custom downloader script.
2. **Landmark extraction** — MediaPipe Hands tracks up to 2 hands per frame, returning 21 3D landmarks per hand (126 features total: x, y, z x 2 hands x 21 points). Landmarks are normalized per hand (centered on the wrist, scaled by wrist-to-middle-finger-MCP distance) for position and scale invariance.
3. **Sequence windowing** — Frames are sampled/padded into fixed-length windows (30 frames) representing one sign in motion.
4. **Model** — A stacked LSTM network trained on normalized landmark sequences classifies each window into one of the 40 vocabulary words (plus an idle/no-sign class).
5. **Browser inference** — The trained Keras model will be converted to TensorFlow.js and run directly in-browser via @mediapipe/hands (JS) for landmark extraction and @tensorflow/tfjs for classification, using a sliding window for continuous prediction.

## Project Structure

    asl-translator/
    ├── training/          # Python: WLASL download, landmark extraction, LSTM training
    │   ├── download_wlasl_subset.py   # Filters WLASL metadata and downloads clips for the 40-word vocabulary
    │   ├── extract_landmarks.py       # Extracts, normalizes, and windows hand landmarks into training sequences
    │   ├── requirements.txt
    │   └── venv/           # Local virtual environment (not committed)
    ├── frontend/          # React app: webcam capture, MediaPipe Hands (JS), TF.js inference
    ├── models/            # Exported TensorFlow.js model artifacts
    ├── data/
    │   ├── raw_videos/    # Downloaded WLASL clips per word (not committed)
    │   └── processed/     # X.npy, y.npy, labels.json — extracted landmark sequences (not committed)
    └── README.md

## Tech Stack

- **Training**: Python 3.11, TensorFlow 2.18.1, mediapipe-silicon 0.9.2.1 (Apple Silicon build), OpenCV (headless), NumPy 1.26.4, protobuf 3.20.3
- **Frontend**: React, TensorFlow.js, @mediapipe/hands
- **Deployment**: Static hosting (Vercel/Netlify) — no backend required, all inference is client-side

## Setup

### Requirements

- Python 3.11 (required — this stack's exact dependency versions are only verified against 3.11; see Known Setup Issues below)
- Node.js and npm (for frontend, once scaffolded)

### Training environment

Create and activate a virtual environment pinned to Python 3.11:

    cd training
    python3.11 -m venv venv
    source venv/bin/activate      # Windows: venv\Scripts\activate

Install dependencies (install from the full requirements file in one command — installing packages individually can cause pip to silently drift version pins):

    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt

Verify the install:

    python -c "import mediapipe as mp; import tensorflow as tf; import cv2; import numpy; print(mp.solutions.hands); print(tf.__version__, cv2.__version__, numpy.__version__)"

### Frontend environment

    cd frontend
    npm install
    npm start

## Known Setup Issues (macOS Apple Silicon)

This stack has several interlocking version constraints that are easy to break if packages are upgraded individually. Documenting them here since they took real troubleshooting to resolve:

- **MediaPipe's official PyPI releases** (0.10.31+) removed the legacy `mp.solutions` API. This project intentionally uses the older Solutions API, so MediaPipe must stay below 0.10.31.
- **MediaPipe 0.10.21 has no arm64 wheel available for this environment.** This project uses the community-maintained `mediapipe-silicon` package instead, which is a drop-in replacement — `import mediapipe as mp` works unchanged in code.
- **Protobuf version must satisfy both TensorFlow and mediapipe-silicon simultaneously.** TensorFlow 2.18.1 requires protobuf >=3.20.3,<6.0.0dev; mediapipe-silicon requires protobuf <4,>=3.11. Pinning to `protobuf==3.20.3` satisfies both.
- **NumPy must stay below 2.0** (`numpy==1.26.4`) — MediaPipe's compiled extensions are not built against NumPy 2.x and will throw `_ARRAY_API not found` / import errors if numpy silently upgrades.
- **Use `opencv-python-headless`, not `opencv-python`.** The standard `opencv-python` package's GUI components conflict with MediaPipe's internally bundled OpenCV libraries on macOS, causing duplicate Objective-C class registration warnings (`CaptureDelegate`, `CVWindow`, etc.). The headless variant avoids this and isn't needed for this project since no OpenCV GUI windows are used.
- **Always install from `requirements.txt` in a single `pip install -r requirements.txt` command**, not package-by-package — installing individually lets pip's resolver silently override earlier pins (numpy in particular).

## Repo Hygiene

This project uses `.gitignore` to exclude the Python `venv/`, `__pycache__/`, editor settings (`.vscode/`), local Python version markers (`.python-version`), raw video and landmark data (`data/raw_videos/`, `data/processed/*.npy`, `data/*.csv`), and `node_modules/` once the frontend is scaffolded. Do not commit the `venv/` folder or any raw/processed data — these should always be regenerated locally via the setup and pipeline scripts.

## Status

🚧 In progress — training environment fully resolved and verified; WLASL data downloaded and landmark extraction complete for all 40 vocabulary words. Next step: build and train the LSTM classifier.

## License

MIT