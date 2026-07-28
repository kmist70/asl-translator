import numpy as np
import random
from scipy.interpolate import interp1d

# Constants (replicated from extract_landmarks.py for self-containment)
SEQUENCE_LENGTH = 30
NUM_HANDS = 2
NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS = 3  # x, y, z
NUM_FEATURES = NUM_HANDS * NUM_LANDMARKS_PER_HAND * NUM_COORDS # 126

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Augments a single landmark sequence (SEQUENCE_LENGTH, NUM_FEATURES) using rotation,
    scaling, noise, temporal jitter, and time warping.

    This function is designed to introduce variability into the training data to improve model generalization
    and combat overfitting, especially with limited real data.

    Augmentation Steps:
    1.  **Random Rotation (XY plane around wrist):**
        Rotates hand landmarks around the wrist (landmark 0) in the XY plane by a
        random angle between -20 and +20 degrees.
    2.  **Random Isotropic Scale Jitter:**
        Multiplies all coordinates by a random factor between 0.9 and 1.1 to simulate
        changes in distance from the camera.
    3.  **Small Additive Gaussian Noise:**
        Adds Gaussian noise (mean 0, std dev 0.015-0.025) to all coordinates to
        simulate sensor noise.
    4.  **Temporal Jitter (Frame Drop/Duplication):**
        With a 50% probability, 1 or 2 frames are randomly dropped or duplicated.
        This simulates minor hesitations or speed variations within the sign itself,
        before the global time warp is applied.
    5.  **Optional Time Warping:**
        With a 50% probability, the sequence is randomly stretched or compressed by up to
        +/- 10% of its original length, then resampled back to SEQUENCE_LENGTH using
        linear interpolation. This simulates variations in signing speed.

    Args:
        sequence (np.ndarray): A NumPy array of shape (SEQUENCE_LENGTH, NUM_FEATURES)
                               representing a single landmark sequence.

    Returns:
        np.ndarray: An augmented NumPy array of the same shape.
    """
    augmented_sequence = sequence.copy() # Work on a copy to avoid modifying original data

    # Reshape for easier manipulation: (frames, hands, landmarks, coords)
    reshaped_sequence = augmented_sequence.reshape(SEQUENCE_LENGTH, NUM_HANDS, NUM_LANDMARKS_PER_HAND, NUM_COORDS)

    # --- 1. Random Rotation (XY plane around wrist) ---
    angle = random.uniform(-20, 20) # degrees
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ])

    for frame_idx in range(SEQUENCE_LENGTH):
        for hand_idx in range(NUM_HANDS):
            hand_landmarks = reshaped_sequence[frame_idx, hand_idx]
            # Only apply rotation if the hand is detected (not all zeros)
            if not np.all(hand_landmarks == 0):
                wrist_coords = hand_landmarks[0, :NUM_COORDS] # Landmark 0 is the wrist

                # Center landmarks around the wrist
                centered_landmarks = hand_landmarks - wrist_coords

                # Apply rotation to the centered landmarks
                # np.dot(A, B.T) is equivalent to A @ B.T for 2D arrays
                rotated_centered_landmarks = np.dot(centered_landmarks, rotation_matrix.T)

                # Translate back
                reshaped_sequence[frame_idx, hand_idx] = rotated_centered_landmarks + wrist_coords

    # --- 2. Random Isotropic Scale Jitter ---
    scale_factor = random.uniform(0.9, 1.1)
    reshaped_sequence *= scale_factor

    # --- 3. Small Additive Gaussian Noise ---
    noise_std = random.uniform(0.015, 0.025)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=reshaped_sequence.shape)
    reshaped_sequence += noise

    # --- 4. Temporal Jitter (Frame Drop/Duplication) ---
    if random.random() < 0.5: # 50% chance to apply jitter
        num_frames_to_jitter = random.randint(1, 2)
        jittered_frames = list(range(SEQUENCE_LENGTH))

        for _ in range(num_frames_to_jitter):
            idx_to_jitter = random.randint(1, SEQUENCE_LENGTH - 2) # Avoid first/last frame
            action = random.choice(['drop', 'duplicate'])

            if action == 'drop' and len(jittered_frames) > num_frames_to_jitter:
                # Ensure we don't drop too many frames
                jittered_frames.pop(idx_to_jitter)
            elif action == 'duplicate':
                jittered_frames.insert(idx_to_jitter, jittered_frames[idx_to_jitter])

        # Resample the jittered sequence back to the original length
        jittered_sequence = reshaped_sequence[jittered_frames, :, :, :]
        # Use interpolation to resize back to SEQUENCE_LENGTH
        resampler = interp1d(np.linspace(0, 1, len(jittered_frames)), jittered_sequence, axis=0)
        reshaped_sequence = resampler(np.linspace(0, 1, SEQUENCE_LENGTH))

    # --- 5. Optional Time Warping ---
    if random.random() < 0.5: # Apply time warping with 50% probability
        stretch_factor = random.uniform(0.9, 1.1) # Stretch/compress by up to +/- 10%
        
        # Calculate new number of frames, ensuring it's at least 2 for interpolation
        # interp1d requires at least 2 points for the x-axis (original_indices)
        new_num_frames = max(2, int(SEQUENCE_LENGTH * stretch_factor))
        
        # Original time points (indices)
        original_indices = np.arange(SEQUENCE_LENGTH)
        # New time points for the warped sequence
        new_indices = np.linspace(0, SEQUENCE_LENGTH - 1, new_num_frames)

        # Flatten the sequence for interpolation across frames for each feature
        # Shape becomes (SEQUENCE_LENGTH, NUM_FEATURES)
        flat_sequence_features = reshaped_sequence.reshape(SEQUENCE_LENGTH, NUM_FEATURES)

        # Perform interpolation
        # interp1d requires at least 2 data points for original_indices
        if SEQUENCE_LENGTH > 1:
            interpolator = interp1d(original_indices, flat_sequence_features, kind='linear', axis=0, fill_value="extrapolate")
            warped_sequence_flat = interpolator(new_indices)
        else: # Fallback for SEQUENCE_LENGTH = 1 (though it's 30 here)
            warped_sequence_flat = np.tile(flat_sequence_features[0], (new_num_frames, 1))

        # Resample the warped sequence back to SEQUENCE_LENGTH
        final_resampled_sequence = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES))
        if new_num_frames > 1:
            # Create new time points for resampling the warped sequence back to original length
            resampler_time_points = np.linspace(0, 1, new_num_frames)
            final_resampler = interp1d(resampler_time_points, warped_sequence_flat, kind='linear', axis=0, fill_value="extrapolate")
            final_resampled_sequence = final_resampler(np.linspace(0, 1, SEQUENCE_LENGTH))
        elif new_num_frames == 1: # If only one frame after warping, duplicate it
            final_resampled_sequence = np.tile(warped_sequence_flat[0], (SEQUENCE_LENGTH, 1))
        
        # Reshape back to (frames, hands, landmarks, coords)
        reshaped_sequence = final_resampled_sequence.reshape(SEQUENCE_LENGTH, NUM_HANDS, NUM_LANDMARKS_PER_HAND, NUM_COORDS)

    return reshaped_sequence.reshape(SEQUENCE_LENGTH, NUM_FEATURES) # Flatten back to (30, 126)