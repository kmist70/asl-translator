# ASL Translator

A real-time American Sign Language (ASL) word and phrase translator that runs entirely in the browser. Hand gestures are captured via webcam, tracked using MediaPipe Hands, and classified by an LSTM sequence model converted to TensorFlow.js for fully client-side inference.

## Overview

This project recognizes a fixed vocabulary of 40 common ASL words from live webcam video, using hand landmark tracking rather than raw video classification. No video data ever leaves the user's browser — all inference runs client-side.

## Vocabulary (v1 — 40 words)

hello, thank you, please, sorry, yes, no, help, want, need, like, love, more, finish, again, learn, understand, know, name, my, you, friend, family, mom, dad, home, school, work, eat, drink, water, hungry, happy, sad, sick, tired, good, bad, where, what, who

## Architecture

1. **Landmark extraction** — MediaPipe Hands tracks up to 2 hands per frame, returning 21 3D landmarks per hand (126 features total: x, y, z x 2 hands x 21 points).
2. **Sequence windowing** — Frames are buffered into fixed-length windows (30 frames) representing one sign in motion.
3. **Model** — A stacked LSTM network trained on normalized landmark sequences classifies each window into one of the 40 vocabulary words (plus an idle/no-sign class).
4. **Browser inference** — The trained Keras model is converted to TensorFlow.js and runs directly in-browser via @mediapipe/hands (JS) for landmark extraction and @tensorflow/tfjs for classification, using a sliding window for continuous prediction.

## Project Structure

    asl-translator/
    ├── training/          # Python: landmark extraction, sequence prep, LSTM training
    ├── frontend/          # React app: webcam capture, MediaPipe Hands (JS), TF.js inference
    ├── models/            # Exported TensorFlow.js model artifacts
    ├── data/              # Landmark sequence datasets (raw video/data excluded via .gitignore)
    └── README.md

## Tech Stack

- **Training**: Python, TensorFlow/Keras, MediaPipe, OpenCV, NumPy
- **Frontend**: React, TensorFlow.js, @mediapipe/hands
- **Deployment**: Static hosting (Vercel/Netlify) — no backend required, all inference is client-side

## Setup

### Training environment

    cd training
    pip install -r requirements.txt

### Frontend environment

    cd frontend
    npm install
    npm start

## Status

🚧 In progress — currently scaffolding project structure and vocabulary dataset collection.

## License

MIT
