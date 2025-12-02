# Heart Rate Prediction from Fingertip Video

This project predicts heart rate and SpO2 from fingertip video data using deep learning and machine learning models.

## Project Structure

- `dataset/`:  he data is from [MEDVSE repository](https://github.com/MahdiFarvardin/MEDVSE).
- `src/`: Contains the Python source code for data loading, preprocessing, modeling, training, evaluation, and prediction.
- `notebooks/`: Can be used for experimental notebooks.
- `main.py`: The main script to run the entire pipeline.
- `requirements.txt`: The required Python packages.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data:**
   - The data is from [MEDVSE repository](https://github.com/MahdiFarvardin/MEDVSE).

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Models

This project implements and compares the following models:

- Random Forest
- 1D Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
