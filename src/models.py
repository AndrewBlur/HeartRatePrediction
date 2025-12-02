from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.ensemble import RandomForestRegressor
from src.utils import coeff_determination, rmse_metric

def build_cnn_model(input_shape):
    model = Sequential([
        # Conv Layer 1
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        # Conv Layer 2
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        # Flatten and Dense
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(2) # Output: HR and SpO2
    ])
    model.compile(optimizer='adam', loss='mse',
                  metrics=['mae', rmse_metric, coeff_determination])
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        # LSTM Layer 1 (Return sequences = True to stack LSTMs)
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),

        # LSTM Layer 2
        LSTM(64, return_sequences=False), # False because we feed into Dense next
        Dropout(0.2),

        # Dense Output
        Dense(32, activation='relu'),
        Dense(2) # Output: HR and SpO2
    ])

    # Using the same custom metrics
    model.compile(optimizer='adam', loss='mse',
                  metrics=['mae', rmse_metric, coeff_determination])
    return model

def build_rf_model(n_estimators=50, max_depth=10, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
