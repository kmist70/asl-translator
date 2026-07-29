# ASL Translator

A real-time American Sign Language (ASL) word translator that runs entirely in the browser. Hand gestures are captured via webcam, tracked using MediaPipe Hands, and classified by an LSTM sequence model converted to TensorFlow.js for fully client-side inference.

## Overview

This project recognizes a fixed vocabulary of 40 common ASL words from live webcam video using hand landmark tracking rather than raw video classification. No video data ever leaves the user's browser — all inference runs client-side.

## Vocabulary

hello, thank you, please, sorry, yes, no, help, want, need, like, love, more, finish, again, learn, understand, know, name, my, you, friend, family, mom, dad, home, school, work, eat, drink, water, hungry, happy, sad, sick, tired, good, bad, where, what, who

## Architecture

1. **Data source** — Video clips are sourced from the WLASL (Word-Level American Sign Language) dataset, filtered down to this project's 40-word vocabulary.
2. **Landmark extraction** — MediaPipe Hands tracks up to 2 hands per frame, returning 21 3D landmarks per hand (126 features total per frame). Hands are indexed in raw detection order from `results.multi_hand_landmarks` (not reordered by handedness label): hand 0 → features[0:63], hand 1 → features[63:126].
3. **Normalization** — For each hand, landmarks are centered on the wrist and scaled by wrist-to-middle-finger-MCP distance for translation and scale invariance.
4. **Sequence windowing** — Each sign sample is represented as a fixed 30-frame landmark sequence.
5. **Model** — A stacked LSTM classifier is trained on normalized landmark sequences. Both LSTM layers use `unroll=True`, which is required for clean TensorFlow.js graph conversion.
6. **Browser inference** — The trained model is exported from Keras, converted to TensorFlow.js, and loaded in-browser for client-side inference via synchronous `model.predict()`.

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
- ~22 classes score 0.00 precision/recall on the test set — expected for thin-data words (e.g. "dad", "hello", "love", "thank you") with as few as 5 real samples before augmentation.

This is well above the 40-class random baseline and confirms the model is learning meaningful temporal structure from hand landmarks, though class imbalance means results should be treated as a baseline, not a finished model.

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

## Frontend

Built with React + Vite. Core pieces:

- `useHandTracker.js` — wraps MediaPipe Hands + Camera utils; normalizes landmarks per hand (center on wrist, scale by wrist-to-middle-finger-MCP distance), flattens to `Float32Array(63)`, zero-fills missing hands with epsilon `1e-6` to avoid divide-by-zero.
- `usePredictor.js` — loads the TF.js model and runs synchronous `model.predict()` + `dataSync()` on buffered 30-frame windows.
- `HandTracker.jsx` — renders webcam video with a canvas overlay drawing landmarks/connectors.
- `PredictionPanel.jsx` — displays current prediction output.
- `App.jsx` — composes `HandTracker` and `PredictionPanel`.

**Known issues resolved this session:**

- `@mediapipe/hands`, `@mediapipe/camera_utils`, `@mediapipe/drawing_utils` are UMD-style packages without proper ESM named exports, which breaks under Vite. Fixed with namespace imports and fallback chains (e.g. `HandsModule.Hands || HandsModule.default?.Hands || window.Hands`).
- `vite.config.js` needed `optimizeDeps.exclude` for all three `@mediapipe/*` packages.
- Fixed an undefined `mpHands` reference left over from a refactor in `useHandTracker.js`.
- Removed a stray duplicate root-level `package.json` that only listed `@tensorflow/tfjs`, keeping the correct `frontend/package.json`.

**Open items before live predictions can be trusted:**

- Label array in `usePredictor.js` is still a placeholder (alphabetical) — must be replaced with the true class order from `data/processed/labels.json` before evaluating prediction quality.
- Full pipeline (camera → landmarks → normalization → 30-frame buffer → model → label mapping) has not yet been live-tested with real signing. Plan is to start with data-rich classes like "drink" (14 samples) before thin-data words.
- Unverified: whether `@mediapipe/hands` JS API's cross-frame hand identity/tracking behavior matches the Python `static_image_mode=False` extraction behavior. A mismatch could cause hand-slot swapping between frames for two-handed signs, since hands are assigned by raw detection order rather than handedness label.

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
│   ├── src/
│   │   ├── hooks/
│   │   │   ├── useHandTracker.js
│   │   │   └── usePredictor.js
│   │   ├── components/
│   │   │   ├── HandTracker.jsx
│   │   │   └── PredictionPanel.jsx
│   │   └── App.jsx
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
- **Frontend**: React + Vite, TensorFlow.js, MediaPipe Hands
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

### Frontend environment

```bash
cd frontend
npm install
npm run dev
```

MediaPipe's `@mediapipe/hands`, `@mediapipe/camera_utils`, and `@mediapipe/drawing_utils` must be excluded from Vite's dependency pre-bundling in `vite.config.js` (`optimizeDeps.exclude`) due to their UMD packaging.

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
✅ React frontend scaffolded (hand tracking, landmark overlay, predictor hook wired up)
✅ MediaPipe/Vite bundling issues resolved
🚧 Label mapping in frontend still placeholder — needs `labels.json` wired in
🚧 End-to-end live prediction not yet validated with real signing
🚧 Hand-slot ordering consistency between Python training and JS inference not yet verified

## License

MIT