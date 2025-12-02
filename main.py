from src.data_loader import load_mths_data
from src.preprocessing import process_data
from src.eda import inspect_data, perform_eda
from src.models import build_cnn_model, build_lstm_model, build_rf_model
from src.training import train_cnn_model, train_lstm_model, train_rf_model
from src.evaluation import evaluate_models
from src.prediction import HeartRateMonitor
from sklearn.model_selection import train_test_split
import os

def main():
    # Configuration
    DATA_PATH = './dataset' 

    # --- Data Loading ---
    # The original notebook cloned a git repo. We will assume the data is in the `dataset` folder.
    # You will need to manually download the data from https://github.com/MahdiFarvardin/MEDVSE
    # and place the 'Data' folder contents into the `dataset` directory.
    
    # Check if the dataset directory is present, if not create it
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory: {DATA_PATH}")
        print("Please download the data from https://github.com/MahdiFarvardin/MEDVSE and place the contents of the 'Data' folder into the 'dataset' directory.")
        return
        
    raw_data = load_mths_data(DATA_PATH)
    if not raw_data:
        print("No data loaded. Please check the dataset path and contents.")
        return
    print("Data Loading Complete.")

    # --- Data Inspection ---
    inspect_data(raw_data)

    # --- Data Processing ---
    X_processed, y_processed = process_data(raw_data)
    print(f"Processed X Shape: {X_processed.shape}")
    print(f"Processed Y Shape: {y_processed.shape}")

    # --- Exploratory Data Analysis ---
    perform_eda(X_processed, y_processed)

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Training Data: {X_train.shape}")
    print(f"Testing Data: {X_test.shape}")

    # --- Model Training ---
    # Build models
    cnn_model = build_cnn_model((30, 3))
    lstm_model = build_lstm_model((30, 3))
    rf_model = build_rf_model()

    # Train models
    cnn_model, _ = train_cnn_model(cnn_model, X_train, y_train, X_test, y_test)
    lstm_model, _ = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)
    rf_model = train_rf_model(rf_model, X_train, y_train)
    
    models = {'cnn': cnn_model, 'lstm': lstm_model, 'rf': rf_model}

    # --- Model Evaluation ---
    evaluate_models(models, X_test, y_test)

    # --- Prediction Example ---
    # Create a dummy video file for testing if it does not exist
    video_path = 'video.mp4'
    if not os.path.exists(video_path):
        print(f"'{video_path}' not found. A dummy video will not be created. Please provide a video for prediction.")
    else:
        app = HeartRateMonitor(model=rf_model, model_type='rf', fs=30)
        hr, spo2 = app.predict(video_path)
        print(f"Predicted Heart Rate: {hr}")
        print(f"Predicted SpO2: {spo2}")


if __name__ == '__main__':
    main()
