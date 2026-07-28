# requirements: mediapipe, opencv-python, numpy, tqdm
# pip install mediapipe opencv-python numpy tqdm

import json
import pathlib
import statistics
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# --- Configuration ---

# Paths
BASE_DIR = pathlib.Path(__file__).parent.parent
RAW_VIDEOS_DIR = BASE_DIR / "data" / "raw_videos"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Sequence & Landmark Configuration
SEQUENCE_LENGTH = 30  # Number of frames per sequence (clip)
NUM_HANDS = 2
NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS = 3  # x, y, z

# MediaPipe Hands model setup
mp_hands = mp.solutions.hands


def normalize_hand_landmarks(hand_landmarks):
    """
    Normalizes hand landmarks to be translation and scale invariant.

    This function is critical for model performance and must be precisely
    replicated in the frontend JavaScript for live inference.

    Normalization Steps:
    1.  **Translation Invariance (Centering):**
        The coordinates of the wrist (landmark 0) are subtracted from all other
        landmarks. This effectively sets the wrist as the origin (0, 0, 0) for
        the hand, making the data robust to where the hand appears in the frame.

    2.  **Scale Invariance (Normalization):**
        All landmark coordinates are divided by the Euclidean distance between
        the wrist (landmark 0) and the middle finger's MCP joint (landmark 9).
        This makes the data robust to the hand's distance from the camera (i.e.,
        its apparent size). A small epsilon is added to the denominator to
        prevent division by zero in rare cases.

    Args:
        hand_landmarks (list): A list of 21 landmark objects from MediaPipe,
                               each with .x, .y, and .z attributes.

    Returns:
        np.ndarray: A flattened NumPy array of shape (63,) containing the
                    normalized (x, y, z) coordinates for the 21 landmarks.
                    Returns a zero array of the same shape if input is invalid.
    """
    if not hand_landmarks:
        return np.zeros(NUM_LANDMARKS_PER_HAND * NUM_COORDS)

    # Convert landmarks to a NumPy array for vectorized operations
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # 1. Center the landmarks around the wrist (landmark 0)
    wrist_coords = coords[0]
    centered_coords = coords - wrist_coords

    # 2. Scale-normalize based on the distance between wrist and middle finger MCP
    wrist_to_middle_mcp = centered_coords[9]
    scale_factor = np.linalg.norm(wrist_to_middle_mcp)

    # Avoid division by zero if the hand is not detected properly
    if scale_factor < 1e-6:
        return np.zeros(NUM_LANDMARKS_PER_HAND * NUM_COORDS)

    normalized_coords = centered_coords / scale_factor

    return normalized_coords.flatten()


def extract_frame_landmarks(frame, hands_model):
    """
    Extracts, normalizes, and combines landmarks for up to two hands from a single frame.

    Args:
        frame (np.ndarray): The input video frame (BGR).
        hands_model: An initialized MediaPipe Hands model instance.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 1D array of 126 features (63 for each hand).
                          If a hand is missing, its features are zero-padded.
            - bool: True if at least one hand was detected, False otherwise.
    """
    # Convert the BGR image to RGB and process it with MediaPipe Hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)

    frame_landmarks = np.zeros(NUM_HANDS * NUM_LANDMARKS_PER_HAND * NUM_COORDS)
    hands_detected = 0

    if results.multi_hand_landmarks:
        hands_detected = len(results.multi_hand_landmarks)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= NUM_HANDS:
                break  # Only process up to NUM_HANDS

            # Normalize the landmarks for the current hand
            normalized_landmarks = normalize_hand_landmarks(hand_landmarks.landmark)

            # Place the 63 features into the correct slot (first 63 or second 63)
            start_idx = i * NUM_LANDMARKS_PER_HAND * NUM_COORDS
            end_idx = start_idx + NUM_LANDMARKS_PER_HAND * NUM_COORDS
            frame_landmarks[start_idx:end_idx] = normalized_landmarks

    return frame_landmarks, (hands_detected > 0)


def process_video_clip(video_path, hands_model):
    """
    Processes a single video file to extract a fixed-length sequence of landmarks.

    Args:
        video_path (pathlib.Path): Path to the video file.
        hands_model: An initialized MediaPipe Hands model instance.

    Returns:
        tuple: A tuple containing:
            - np.ndarray or None: The processed (30, 126) landmark sequence, or None on failure.
            - float: The percentage of frames with zero hands detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   -> Error: Could not open video file {video_path}")
        return None, 100.0

    all_frame_landmarks = []
    total_frames = 0
    frames_with_no_hands = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        landmarks, hand_detected = extract_frame_landmarks(frame, hands_model)
        all_frame_landmarks.append(landmarks)
        if not hand_detected:
            frames_with_no_hands += 1

    cap.release()

    if not all_frame_landmarks:
        return None, 100.0

    zero_hand_percentage = (frames_with_no_hands / total_frames) * 100 if total_frames > 0 else 100.0

    # Pad or truncate the sequence to SEQUENCE_LENGTH
    num_frames = len(all_frame_landmarks)
    processed_sequence = np.zeros((SEQUENCE_LENGTH, NUM_HANDS * NUM_LANDMARKS_PER_HAND * NUM_COORDS))

    if num_frames > 0:
        if num_frames >= SEQUENCE_LENGTH:
            # Sample evenly spaced frames from the clip
            indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
            for i, idx in enumerate(indices):
                processed_sequence[i] = all_frame_landmarks[idx]
        else:
            # Pad with zeros at the end
            processed_sequence[:num_frames] = np.array(all_frame_landmarks)

    return processed_sequence, zero_hand_percentage


def main():
    """
    Main function to iterate through raw videos, extract landmarks,
    and save the processed data.
    """
    print("Starting landmark extraction process...")
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    # Get all glosses (subdirectories in raw_videos)
    glosses = [d.name for d in RAW_VIDEOS_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(glosses)} glosses (words) to process.")

    # Initialize MediaPipe Hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=NUM_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_sequences = []
    all_labels = []
    quality_report = defaultdict(list)

    for gloss in sorted(glosses):
        gloss_dir = RAW_VIDEOS_DIR / gloss
        video_files = list(gloss_dir.glob("*.mp4"))
        print(f"\nProcessing '{gloss}': {len(video_files)} clips found.")

        if not video_files:
            continue

        # Create output directory for the gloss
        output_gloss_dir = PROCESSED_DATA_DIR / gloss
        output_gloss_dir.mkdir(exist_ok=True)

        for video_path in tqdm(video_files, desc=f"  -> {gloss}", unit="clip"):
            output_npy_path = output_gloss_dir / f"{video_path.stem}.npy"

            sequence, zero_hand_perc = process_video_clip(video_path, hands_model)

            if sequence is not None:
                np.save(output_npy_path, sequence)
                all_sequences.append(sequence)
                all_labels.append(gloss)
                quality_report[gloss].append(zero_hand_perc)
            else:
                tqdm.write(f"   -> Failed to process {video_path}")

    hands_model.close()

    # --- Save combined dataset and labels ---
    print("\nSaving combined dataset files...")
    if all_sequences:
        X = np.array(all_sequences)
        y = np.array(all_labels)
        np.save(PROCESSED_DATA_DIR / "X.npy", X)
        np.save(PROCESSED_DATA_DIR / "y.npy", y)
        print(f"Saved X.npy with shape: {X.shape}")
        print(f"Saved y.npy with shape: {y.shape}")

        # Save label map
        unique_labels = sorted(list(set(all_labels)))
        label_map = {i: label for i, label in enumerate(unique_labels)}
        with open(PROCESSED_DATA_DIR / "labels.json", "w") as f:
            json.dump(label_map, f, indent=4)
        print("Saved labels.json mapping.")

    # --- Print final quality report ---
    print("\n--- Data Quality Report ---")
    print(f"{'Word':<15} | {'Clips':>5} | {'Avg % No Hands':>15} | {'Status'}")
    print("-" * 60)
    for gloss in sorted(quality_report.keys()):
        clips_processed = len(quality_report[gloss])
        if clips_processed > 0:
            avg_no_hands = statistics.mean(quality_report[gloss])
            status = "LOW QUALITY" if avg_no_hands > 30.0 else "OK"
            print(f"{gloss:<15} | {clips_processed:>5} | {avg_no_hands:>14.1f}% | {status}")
        else:
            print(f"{gloss:<15} | {'0':>5} | {'N/A':>15} | NO CLIPS")
    print("-" * 60)
    print("Landmark extraction complete.")


if __name__ == "__main__":
    main()