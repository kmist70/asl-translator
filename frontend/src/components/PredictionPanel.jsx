import React from 'react';

/**
 * PredictionPanel component displays the top 3 predicted words and their confidence scores.
 *
 * @param {object} props
 * @param {Array<Object>} props.predictions - An array of prediction objects, e.g.,
 *                                            [{ word: 'hello', confidence: 0.95 }, ...]
 */
const PredictionPanel = ({ predictions }) => {
    return (
        <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', marginTop: '20px' }}>
            <h3>Top 3 Predictions:</h3>
            {predictions.length > 0 ? (
                <ul>
                    {predictions.map((p, index) => (
                        <li key={index}>{p.word}: {(p.confidence * 100).toFixed(2)}%</li>
                    ))}
                </ul>
            ) : (<p>Waiting for predictions...</p>)}
        </div>
    );
};

export default PredictionPanel;