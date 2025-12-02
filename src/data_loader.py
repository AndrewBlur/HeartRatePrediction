import os
import numpy as np
import pandas as pd
import glob

# Configuration
DATA_PATH = './MEDVSE/MTHS/Data'  # Change this to your actual path
FS_SIGNAL = 30  # Signal sampling rate
FS_LABEL = 1    # Label sampling rate

def load_mths_data(path):
    subjects_data = []

    # specific logic: finding matching signal and label files
    # Assuming files are named like 'signal_1.npy', 'label_1.npy'
    signal_files = sorted(glob.glob(os.path.join(path, 'signal_*.npy')))

    for sig_file in signal_files:
        # Construct label filename based on signal filename
        base_name = os.path.basename(sig_file)
        patient_id = base_name.split('_')[1].split('.')[0] # Extracts 'x' from signal_x.npy
        lbl_file = os.path.join(path, f'label_{patient_id}.npy')

        if os.path.exists(lbl_file):
            signal = np.load(sig_file) # Expected shape: (N, 3) or (3, N) - we will check in inspection
            label = np.load(lbl_file)  # Expected shape: (M, 2)
            subjects_data.append({'id': patient_id, 'signal': signal, 'label': label})
        else:
            print(f"Warning: Label file not found for {sig_file}")

    return subjects_data
