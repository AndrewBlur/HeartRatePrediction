import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def inspect_data(data):
    print("--- Data Inspection ---")

    # 1. Check number of subjects
    print(f"Total Subjects Loaded: {len(data)}")

    # 2. Inspect first subject details
    if len(data) > 0:
        first_sub = data[0]
        print(f"Subject {first_sub['id']} Signal Shape: {first_sub['signal'].shape}")
        print(f"Subject {first_sub['id']} Label Shape: {first_sub['label'].shape}")

        # 3. Check Signal Stats
        print(f"Signal (RGB) Mean: {np.mean(first_sub['signal'], axis=0)}")
        print(f"Signal (RGB) Min: {np.min(first_sub['signal'], axis=0)}")

        # 4. Check Label distribution
        print(f"Label (HR, SpO2) Mean: {np.mean(first_sub['label'], axis=0)}")

def perform_eda(X, y):
    print("--- Exploratory Data Analysis ---")

    # 1. Target Variable Distribution (HR)
    plt.figure(figsize=(10, 4))
    sns.histplot(y[:, 0], kde=True, color='red')
    plt.title('Distribution of Heart Rates (Ground Truth)')
    plt.xlabel('BPM')
    plt.show()

    # 2. RGB Signal Visualization (One 30-frame window)
    sample_idx = 0
    plt.figure(figsize=(10, 4))
    plt.plot(X[sample_idx, :, 0], color='r', label='Red')
    plt.plot(X[sample_idx, :, 1], color='g', label='Green')
    plt.plot(X[sample_idx, :, 2], color='b', label='Blue')
    plt.title(f'RGB Signal for 1 Second (Label HR: {y[sample_idx, 0]:.1f})')
    plt.legend()
    plt.show()

    # 3. Correlation between HR and SpO2
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y[:, 0], y=y[:, 1])
    plt.title('Correlation: HR vs SpO2')
    plt.xlabel('Heart Rate')
    plt.ylabel('SpO2')
    plt.show()
