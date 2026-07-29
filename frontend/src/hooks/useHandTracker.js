import { useState, useEffect, useRef, useCallback } from 'react';
import * as HandsModule from '@mediapipe/hands';
import * as CameraUtilsModule from '@mediapipe/camera_utils';
import * as DrawingUtilsModule from '@mediapipe/drawing_utils';

const drawConnectors = DrawingUtilsModule.drawConnectors || DrawingUtilsModule.default?.drawConnectors || window.drawConnectors;
const drawLandmarks = DrawingUtilsModule.drawLandmarks || DrawingUtilsModule.default?.drawLandmarks || window.drawLandmarks;

const Hands = HandsModule.Hands || HandsModule.default?.Hands || window.Hands;
const Camera = CameraUtilsModule.Camera || CameraUtilsModule.default?.Camera || window.Camera;
const mpHands = HandsModule;

// --- Configuration ---
const NUM_HANDS = 2;
const NUM_LANDMARKS_PER_HAND = 21;
const NUM_COORDS = 3; // x, y, z
const NUM_FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * NUM_COORDS; // 63
const TOTAL_FEATURES_PER_FRAME = NUM_HANDS * NUM_FEATURES_PER_HAND; // 126

/**
 * Normalizes hand landmarks to be translation and scale invariant, matching the Python training pipeline.
 *
 * Normalization Steps:
 * 1.  **Translation Invariance (Centering):**
 *     The coordinates of the wrist (landmark 0) are subtracted from all other
 *     landmarks. This effectively sets the wrist as the origin (0, 0, 0) for
 *     the hand.
 * 2.  **Scale Invariance (Normalization):**
 *     All landmark coordinates are divided by the Euclidean distance between
 *     the wrist (landmark 0) and the middle finger's MCP joint (landmark 9),
 *     computed post-centering.
 * 3.  **Flattening:**
 *     The 21 (x, y, z) coordinates are flattened into a 63-element Float32Array.
 *
 * @param {Array<Object>} handLandmarks - A list of 21 landmark objects from MediaPipe,
 *                                        each with .x, .y, and .z attributes.
 * @returns {Float32Array} A flattened Float32Array of shape (63,) containing the
 *                         normalized (x, y, z) coordinates for the 21 landmarks.
 *                         Returns a zero array of the same shape if input is invalid or scale is too small.
 */
const normalizeHandLandmarks = (handLandmarks) => {
    if (!handLandmarks || handLandmarks.length === 0) {
        return new Float32Array(NUM_FEATURES_PER_HAND).fill(0);
    }

    // Convert landmarks to a flat array for easier manipulation
    const coords = new Float32Array(NUM_LANDMARKS_PER_HAND * NUM_COORDS);
    for (let i = 0; i < NUM_LANDMARKS_PER_HAND; i++) {
        coords[i * NUM_COORDS] = handLandmarks[i].x;
        coords[i * NUM_COORDS + 1] = handLandmarks[i].y;
        coords[i * NUM_COORDS + 2] = handLandmarks[i].z;
    }

    // 1. Center the landmarks around the wrist (landmark 0)
    const wristX = coords[0];
    const wristY = coords[1];
    const wristZ = coords[2];

    const centeredCoords = new Float32Array(NUM_FEATURES_PER_HAND);
    for (let i = 0; i < NUM_LANDMARKS_PER_HAND; i++) {
        centeredCoords[i * NUM_COORDS] = coords[i * NUM_COORDS] - wristX;
        centeredCoords[i * NUM_COORDS + 1] = coords[i * NUM_COORDS + 1] - wristY;
        centeredCoords[i * NUM_COORDS + 2] = coords[i * NUM_COORDS + 2] - wristZ;
    }

    // 2. Scale-normalize based on the distance between wrist (0) and middle finger MCP (9)
    // The distance is computed from the *centered* coordinates.
    const middleMcpX = centeredCoords[9 * NUM_COORDS];
    const middleMcpY = centeredCoords[9 * NUM_COORDS + 1];
    const middleMcpZ = centeredCoords[9 * NUM_COORDS + 2];

    const scaleFactor = Math.sqrt(
        middleMcpX * middleMcpX +
        middleMcpY * middleMcpY +
        middleMcpZ * middleMcpZ
    );

    // Avoid division by zero if the hand is not detected properly or is too small
    if (scaleFactor < 1e-6) {
        return new Float32Array(NUM_FEATURES_PER_HAND).fill(0);
    }

    const normalizedCoords = new Float32Array(NUM_FEATURES_PER_HAND);
    for (let i = 0; i < NUM_FEATURES_PER_HAND; i++) {
        normalizedCoords[i] = centeredCoords[i] / scaleFactor;
    }

    return normalizedCoords;
};

export const useHandTracker = () => {
    const videoRef = useRef(null);
    const handsRef = useRef(null);
    const cameraRef = useRef(null);

    const [latestFrameFeatures, setLatestFrameFeatures] = useState(new Float32Array(TOTAL_FEATURES_PER_FRAME).fill(0));
    const [rawMultiHandLandmarks, setRawMultiHandLandmarks] = useState([]);
    const [rawMultiHandedness, setRawMultiHandedness] = useState([]);

    const onResults = useCallback((results) => {
        const frameFeatures = new Float32Array(TOTAL_FEATURES_PER_FRAME).fill(0);
        const currentRawLandmarks = [];
        const currentRawHandedness = [];

        if (results.multiHandLandmarks && results.multiHandedness) {
            for (let i = 0; i < Math.min(results.multiHandLandmarks.length, NUM_HANDS); i++) {
                const handLandmarks = results.multiHandLandmarks[i];
                const handedness = results.multiHandedness[i];

                const normalizedFeatures = normalizeHandLandmarks(handLandmarks);
                frameFeatures.set(normalizedFeatures, i * NUM_FEATURES_PER_HAND);

                currentRawLandmarks.push(handLandmarks);
                currentRawHandedness.push(handedness);
            }
        }

        setLatestFrameFeatures(frameFeatures);
        setRawMultiHandLandmarks(currentRawLandmarks);
        setRawMultiHandedness(currentRawHandedness);
    }, []);

    useEffect(() => {
        handsRef.current = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        handsRef.current.setOptions({
            maxNumHands: NUM_HANDS,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        handsRef.current.onResults(onResults);

        if (videoRef.current) {
            cameraRef.current = new Camera(videoRef.current, {
                onFrame: async () => {
                    await handsRef.current.send({ image: videoRef.current });
                },
                width: 1280, // Example resolution, adjust as needed
                height: 720,
            });
            cameraRef.current.start();
        }

        return () => {
            if (cameraRef.current) cameraRef.current.stop();
            if (handsRef.current) handsRef.current.close();
        };
    }, [onResults]);

    /**
     * Hand Ordering Convention:
     * MediaPipe's `results.multiHandLandmarks` array directly corresponds to the order
     * in which hands were detected in the frame. This hook processes them in that order.
     *
     * ASSUMPTION: The Python training pipeline's feature concatenation for two hands
     * (hand1_features + hand2_features) implicitly relies on a consistent ordering.
     * If the training pipeline used MediaPipe's `handedness` label (e.g., always 'Left' then 'Right' hand features)
     * rather than the raw detection order, there will be a silent mismatch here.
     * This implementation uses MediaPipe's raw detection order.
     *
     * To verify, you would need to inspect the Python `extract_landmarks.py` to see
     * how `frame_landmarks` is populated when `results.multi_hand_landmarks` has two entries.
     * If it iterates `results.multi_hand_landmarks` directly, then this implementation matches.
     * If it sorts or reorders based on `handedness`, then this needs adjustment.
     */
    return { videoRef, latestFrameFeatures, rawMultiHandLandmarks, rawMultiHandedness, mpHands };
};