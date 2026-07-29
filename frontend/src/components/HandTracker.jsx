import React, { useEffect, useRef } from 'react';
import * as DrawingUtilsModule from '@mediapipe/drawing_utils';

const drawConnectors = DrawingUtilsModule.drawConnectors || DrawingUtilsModule.default?.drawConnectors || window.drawConnectors;
const drawLandmarks = DrawingUtilsModule.drawLandmarks || DrawingUtilsModule.default?.drawLandmarks || window.drawLandmarks;

/**
 * HandTracker component displays the webcam feed and overlays MediaPipe hand landmarks.
 *
 * @param {object} props
 * @param {React.RefObject<HTMLVideoElement>} props.videoRef - Ref to the video element for webcam feed.
 * @param {Array<Object>} props.rawMultiHandLandmarks - Array of raw landmark data from MediaPipe.
 * @param {Array<Object>} props.rawMultiHandedness - Array of handedness data from MediaPipe.
 * @param {object} props.mpHands - MediaPipe Hands module for constants like HAND_CONNECTIONS.
 */
const HandTracker = ({ videoRef, rawMultiHandLandmarks, rawMultiHandedness, mpHands }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvasElement = canvasRef.current;
        const canvasCtx = canvasElement.getContext('2d');

        const draw = () => {
            if (!videoRef.current || !canvasCtx) return;

            // Set canvas dimensions to match video
            canvasElement.width = videoRef.current.videoWidth;
            canvasElement.height = videoRef.current.videoHeight;

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            // Flip canvas horizontally to mirror the video feed
            canvasCtx.translate(canvasElement.width, 0);
            canvasCtx.scale(-1, 1);

            // Draw landmarks if available
            if (rawMultiHandLandmarks && rawMultiHandLandmarks.length > 0) {
                for (const landmarks of rawMultiHandLandmarks) {
                    drawConnectors(canvasCtx, landmarks, mpHands.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
                    drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
                }
            }
            canvasCtx.restore();
        };

        // Request animation frame to continuously draw
        const animationFrameId = requestAnimationFrame(draw);

        return () => cancelAnimationFrame(animationFrameId);
    }, [videoRef, rawMultiHandLandmarks, rawMultiHandedness, mpHands]);

    return (
        <div style={{ position: 'relative', width: 'fit-content' }}>
            <video ref={videoRef} style={{ width: '100%', height: 'auto', transform: 'scaleX(-1)' }} autoPlay playsInline muted />
            <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
        </div>
    );
};

export default HandTracker;