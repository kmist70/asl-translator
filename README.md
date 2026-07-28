# ASL Translator

A real-time American Sign Language (ASL) word translator that runs entirely in the browser. Hand gestures are captured via webcam, tracked using MediaPipe Hands, and classified by an LSTM sequence model converted to TensorFlow.js for fully client-side inference.

## Overview

This project recognizes a fixed vocabulary of 40 common ASL words from live webcam video using hand landmark tracking rather than raw video classification. No video data ever leaves the user's browser — all inference runs client-side.

## Vocabulary

hello, thank you, please, sorry, yes, no, help, want, need, like, love, more, finish, again, learn, understand, know, name, my, you, friend, family, mom, dad, home, school, work, eat, drink, water, hungry, happy, sad, sick, tired, good, bad, where, what, who

## Architecture

1. **Data source** — Video clips are sourced from the WLASL (Word-Level American Sign Language) dataset, filtered down to this project's 40-word vocabulary.
2. **Landmark extraction** — MediaPipe Hands tracks up to 2 hands per frame, returning 21 3D landmarks per hand (126 features total per frame).
3. **Normalization** — For each hand, landmarks are centered on the wrist and scaled by wrist-to-middle-finger-MCP distance for translation and scale invariance.
4. **Sequence windowing** — Each sign sample is represented as a fixed 30-frame landmark sequence.
5. **Model** — A stacked LSTM classifier is trained on normalized landmark sequences. Both LSTM layers use `unroll=True`, which is required for clean TensorFlow.js graph conversion.
6. **Browser inference** — The trained model is exported from Keras, converted to TensorFlow.js, and loaded in-browser for client-side inference.

## Model

Current architecture:

- LSTM(64, return_sequences=True, unroll=True, kernel_regularizer=l2(0.001))
- Dropout(0.4)
- LSTM(32, unroll=True, kernel_regularizer=l2(0.001))
- Dropout(0.4)
- Dense(16, relu, kernel_regularizer=l2(0.001))
- Dense(40, softmax)

Input shape: `(30, 126)`  
Output shape: `(40,)`

## Results

Dataset summary:

- 359 total samples across 40 classes.
- Per-class real sample counts range from 5 to 14.
- Test split uses 1 sample per class, so per-class precision/recall should be interpreted cautiously.

Latest baseline:

- **Top-1 test accuracy:** 47.5%
- **Top-3 test accuracy:** 67.5%

This is well above the 40-class random baseline and confirms the model is learning meaningful temporal structure from hand landmarks.

## TensorFlow.js deployment status

The TensorFlow.js conversion path is working and verified.

Key lessons from deployment:

- Plain Keras/SavedModel export of the original LSTM caused TF.js graph execution failures due to dynamic control-flow ops.
- Setting `unroll=True` on both LSTM layers fixed the conversion/runtime issue for this fixed-length 30-frame sequence task.
- `model.export(...)` worked better than manual SavedModel signature wrapping.
- The converted TF.js graph model was successfully loaded and run in Node.js.
- Verified inference call:
  ```js
  model.predict({ keras_tensor: inputTensor })
  ```
- Verified output shape: `[1, 40]`
- Verified softmax sum: `0.99999994`

## Project Structure

```text
asl-translator/
├── training/
│   ├── download_wlasl_subset.py
│   ├── extract_landmarks.py
│   ├── augment_landmarks.py
│   ├── train_model.py
│   ├── export_for_tfjs.py
│   ├── requirements.txt
│   ├── results/
│   ├── raw_data/                # gitignored
│   ├── models_checkpoints/      # gitignored checkpoints
│   └── venv/                    # local only, gitignored
├── frontend/
│   └── public/
│       └── model/               # committed TF.js model artifacts
├── models/
│   └── asl_lstm_model/          # regenerable SavedModel, gitignored
├── data/
│   ├── raw_videos/              # gitignored
│   └── processed/               # gitignored
├── README.md
└── LICENSE
```

## Tech Stack

- **Training**: Python 3.11, TensorFlow 2.18.1, mediapipe-silicon 0.9.2.1, NumPy 1.26.4, protobuf 3.20.3, opencv-python-headless 4.11.0.86
- **Frontend**: React + TensorFlow.js + MediaPipe Hands
- **Deployment**: Static hosting, no backend required

## Setup

### Training environment

Use Python 3.11 exactly.

```bash
cd training
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Important dependency notes

- Install with `pip install -r requirements.txt` in one shot, not package-by-package.
- `mediapipe-silicon==0.9.2.1` is required on this Apple Silicon setup.
- `protobuf==3.20.3` is the overlap that works with both TensorFlow 2.18.1 and mediapipe-silicon.
- `numpy` must stay below 2.
- Use `opencv-python-headless==4.11.0.86`, not `4.11.0`.

### TF.js conversion note

TensorFlow.js conversion was done in a separate Python virtual environment because installing `tensorflowjs` directly into the training environment caused dependency conflicts.

## Repo Hygiene

Do not commit:

- `training/venv/`
- `converter_venv/`
- `node_modules/`
- raw videos or processed `.npy` data
- regenerable SavedModel exports and checkpoints

Commit:

- source code
- README
- results plots/reports
- final browser-ready TF.js model artifacts in `frontend/public/model/`

## Status

✅ Data download complete  
✅ Landmark extraction complete  
✅ Augmentation pipeline complete  
✅ LSTM training complete  
✅ Baseline evaluation complete  
✅ Keras export complete  
✅ TensorFlow.js conversion complete  
✅ TF.js inference verified in Node.js  
🚧 React frontend scaffolding and real-time browser UI next

## License

MIT