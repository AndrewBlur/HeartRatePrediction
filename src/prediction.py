import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class HeartRateMonitor:
    def __init__(self, model, model_type='rf', fs=30):
        self.model = model
        self.model_type = model_type
        self.fs = fs
        self.scaler = StandardScaler()

    def extract_signal_from_video(self, video_path):
        """
        Reads a video file and extracts the average RGB value from a
        CENTER REGION OF INTEREST (ROI).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        raw_signal = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- ROI IMPLEMENTATION (Center Crop) ---
            h, w, _ = frame_rgb.shape
            min_dim = min(h, w)
            roi_size = int(min_dim * 0.5)

            center_x, center_y = w // 2, h // 2
            x1 = center_x - roi_size // 2
            x2 = center_x + roi_size // 2
            y1 = center_y - roi_size // 2
            y2 = center_y + roi_size // 2

            roi_frame = frame_rgb[y1:y2, x1:x2]
            # ----------------------------------------

            avg_color = np.mean(roi_frame, axis=(0, 1))
            raw_signal.append(avg_color)

        cap.release()
        return np.array(raw_signal)

    def preprocess_signal(self, raw_signal):
        signal = np.clip(raw_signal, 0, 255)
        # Note: Fitting on the new video signal aligns it to the model's expected distribution
        signal_scaled = self.scaler.fit_transform(signal)
        return signal_scaled

    def segment_signal(self, signal):
        num_seconds = len(signal) // self.fs
        windows = []
        for i in range(num_seconds):
            start_idx = i * self.fs
            end_idx = start_idx + self.fs
            window = signal[start_idx:end_idx, :]
            windows.append(window)
        return np.array(windows)

    def predict(self, video_path):
        print(f"Processing video: {video_path}...")

        # 1. Extract
        raw_signal = self.extract_signal_from_video(video_path)
        if raw_signal is None or len(raw_signal) == 0:
            print("No signal extracted.")
            return

        print(f"Video duration: {len(raw_signal)/self.fs:.1f} seconds")

        # 2. Preprocess
        clean_signal = self.preprocess_signal(raw_signal)

        # 3. Segment
        X_windows = self.segment_signal(clean_signal)
        if len(X_windows) == 0:
            print("Video too short (needs at least 1 second).")
            return

        # 4. Model Prediction
        if self.model_type == 'rf':
            X_input = X_windows.reshape(X_windows.shape[0], -1)
            predictions = self.model.predict(X_input)
        else:
            X_input = X_windows
            predictions = self.model.predict(X_input, verbose=0)

        # 5. Extract HR (Col 0) and SpO2 (Col 1)
        # Ensure predictions shape is (N, 2)
        if predictions.ndim > 1 and predictions.shape[1] >= 2:
            hr_values = predictions[:, 0]
            spo2_values = predictions[:, 1]
        else:
            print("Model output format not recognized. Returning only HR if available.")
            hr_values = predictions[:, 0] if predictions.ndim > 1 else predictions
            spo2_values = np.zeros_like(hr_values)

        # 6. Generate Graph
        self.plot_results(hr_values, spo2_values)

        return hr_values, spo2_values

    def plot_results(self, hr_values, spo2_values):
        time_axis = np.arange(len(hr_values))

        plt.figure(figsize=(12, 8))

        # --- Plot 1: Heart Rate ---
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, hr_values, 'o-', color='r', label='Predicted HR')
        avg_hr = np.mean(hr_values)
        plt.axhline(y=avg_hr, color='darkred', linestyle='--', alpha=0.5, label=f'Avg HR: {avg_hr:.1f}')
        plt.ylabel('BPM')
        plt.title('Heart Rate (BPM) vs Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # --- Plot 2: SpO2 ---
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, spo2_values, 'o-', color='b', label='Predicted SpO2')
        avg_spo2 = np.mean(spo2_values)
        plt.axhline(y=avg_spo2, color='darkblue', linestyle='--', alpha=0.5, label=f'Avg SpO2: {avg_spo2:.1f}%')
        plt.ylabel('SpO2 (%)')
        plt.title('Oxygen Saturation (SpO2) vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylim(80, 100) # SpO2 is typically between 80-100%
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Summary -> Avg HR: {avg_hr:.1f} BPM | Avg SpO2: {avg_spo2:.1f}%")
