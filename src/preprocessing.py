from sklearn.preprocessing import StandardScaler
import numpy as np

FS_SIGNAL = 30  # Signal sampling rate

def process_data(data):
    X_list = []
    y_list = []


    # Scaler for RGB values
    scaler = StandardScaler()

    for subject in data:
        sig = subject['signal'] # (3570, 3)
        lbl = subject['label']  # (119, 2)

        # 1. Handling Outliers/Cleaning (Simple clipping for RGB)
        sig = np.clip(sig, 0, 255)

        # 2. Scaling (Standardization per subject is usually better for rPPG)
        sig = scaler.fit_transform(sig)

        # 3. Windowing Strategy (30 frames -> 1 label)
        # Ensure lengths match
        num_seconds = min(len(sig) // FS_SIGNAL, len(lbl))

        for i in range(num_seconds):
            start_idx = i * FS_SIGNAL
            end_idx = start_idx + FS_SIGNAL

            # Extract 30 frames
            window = sig[start_idx:end_idx, :] # Shape (30, 3)

            # Extract target (HR, SpO2)
            target = lbl[i]

            X_list.append(window)
            y_list.append(target)

    return np.array(X_list), np.array(y_list)
