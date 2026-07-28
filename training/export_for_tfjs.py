from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2

model = keras.Sequential([
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001), unroll=True, input_shape=(30, 126)),
    Dropout(0.4),
    LSTM(32, kernel_regularizer=l2(0.001), unroll=True),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(40, activation='softmax')
])

model.load_weights('training/models_checkpoints/best_model.keras')
model.export('models/asl_lstm_model_v4')