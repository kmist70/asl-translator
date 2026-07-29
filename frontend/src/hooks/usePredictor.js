import { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

// --- Configuration ---
const MODEL_PATH = '/model/model.json';
const SEQUENCE_LENGTH = 30; // Fixed length of input sequence for the model
const NUM_FEATURES = 126;   // 2 hands * 21 landmarks * 3 coords
const NUM_CLASSES = 40;     // Number of output classes from the model
const PREDICTION_THROTTLE_FRAMES = 5; // Run prediction every N frames

// TODO: This MUST be replaced with the exact label order from your
// Python training pipeline's data/processed/labels.json.
// Misalignment here will lead to silently incorrect predictions.
const LABELS = [
    "again", "bad", "dad", "drink", "eat", "family", "finish", "friend",
    "good", "happy", "hello", "help", "home", "hungry", "know", "learn",
    "like", "love", "mom", "more", "my", "name", "need", "no", "please",
    "sad", "school", "sick", "sorry", "thank you", "tired", "understand",
    "want", "water", "what", "where", "who", "work", "yes", "you"
].sort(); // Sorted alphabetically for now, replace with actual order!

export const usePredictor = (latestFrameFeatures) => {
    const modelRef = useRef(null);
    const predictionBufferRef = useRef([]); // Stores Float32Array(126) frames
    const frameCounterRef = useRef(0);

    const [predictions, setPredictions] = useState([]); // [{ word: string, confidence: number }]

    // Load the model once on mount
    useEffect(() => {
        const loadModel = async () => {
            try {
                console.log('Loading TensorFlow.js model...');
                const loadedModel = await tf.loadGraphModel(MODEL_PATH);
                modelRef.current = loadedModel;
                console.log('Model loaded successfully:', loadedModel);
                // Initialize buffer
                predictionBufferRef.current = [];
            } catch (error) {
                console.error('Failed to load TF.js model:', error);
            }
        };
        loadModel();
    }, []);

    // Process new frames from useHandTracker
    useEffect(() => {
        if (!modelRef.current || !latestFrameFeatures || latestFrameFeatures.length === 0) {
            return;
        }

        // Add the new frame to the buffer
        predictionBufferRef.current.push(latestFrameFeatures);

        // Maintain fixed buffer size (FIFO)
        if (predictionBufferRef.current.length > SEQUENCE_LENGTH) {
            predictionBufferRef.current.shift();
        }

        // Throttle predictions
        frameCounterRef.current++;
        if (frameCounterRef.current % PREDICTION_THROTTLE_FRAMES !== 0) {
            return;
        }

        // Only predict if the buffer is full
        if (predictionBufferRef.current.length === SEQUENCE_LENGTH) {
            // Create a 3D tensor from the buffer: [1, SEQUENCE_LENGTH, NUM_FEATURES]
            const inputTensor = tf.tensor3d(
                [predictionBufferRef.current],
                [1, SEQUENCE_LENGTH, NUM_FEATURES],
                'float32'
            );

            let outputTensor;
            try {
                // CONFIRMED MODEL FACT: Use synchronous predict()
                outputTensor = modelRef.current.predict({ 'keras_tensor': inputTensor });
                const outputData = outputTensor.dataSync(); // Get raw probabilities

                // Get top 3 predictions
                const topK = tf.topk(outputTensor, 3);
                const topKIndices = topK.indices.dataSync();
                const topKValues = topK.values.dataSync();

                const currentPredictions = Array.from(topKIndices).map((index, i) => ({
                    word: LABELS[index],
                    confidence: topKValues[i],
                }));
                setPredictions(currentPredictions);

            } catch (error) {
                console.error("Error during prediction:", error);
            } finally {
                // Dispose tensors to prevent memory leaks
                inputTensor.dispose();
                if (outputTensor) outputTensor.dispose();
            }
        }
    }, [latestFrameFeatures]); // Re-run effect when a new frame is available

    return { predictions };
};