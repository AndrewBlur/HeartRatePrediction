import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def plot_enhanced_history(history, model_name):
    # Extract metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    rmse = history.history['rmse_metric']
    val_rmse = history.history['val_rmse_metric']
    r2 = history.history['coeff_determination']
    val_r2 = history.history['val_coeff_determination']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(18, 5))

    # Plot 1: Loss (MSE)
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, 'y', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title(f'{model_name}: Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # Plot 2: RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, rmse, 'b', label='Train RMSE')
    plt.plot(epochs, val_rmse, 'g', label='Val RMSE')
    plt.title(f'{model_name}: RMSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # Plot 3: R2 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, r2, 'purple', label='Train R2')
    plt.plot(epochs, val_r2, 'orange', label='Val R2')
    plt.title(f'{model_name}: R2 Score')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_cnn_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    print("--- Training CNN Model ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    plot_enhanced_history(history, "CNN Model")
    return model, history

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    print("--- Training LSTM Model ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    plot_enhanced_history(history, "LSTM Model")
    return model, history

def train_rf_model(model, X_train, y_train):
    print("Generating Learning Curve for Random Forest... (This may take a moment)")
    # Flatten Data for Non-Deep Learning Model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train_flat,
        y_train,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # Convert negative MAE to positive for plotting
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    # Train Final Model
    model.fit(X_train_flat, y_train)
    print("Baseline Model Trained.")

    # Plot Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Error")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validation Error")
    plt.title("Baseline Learning Curve (Random Forest)")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    return model
