from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_models(models, X_test, y_test):
    # 1. Generate Predictions
    pred_rf = models['rf'].predict(X_test.reshape(X_test.shape[0], -1))
    pred_cnn = models['cnn'].predict(X_test, verbose=0)
    pred_lstm = models['lstm'].predict(X_test, verbose=0)

    # 2. Organize Results
    model_names = ['Random Forest', 'CNN', 'LSTM']

    r2_scores = [
        r2_score(y_test, pred_rf),
        r2_score(y_test, pred_cnn),
        r2_score(y_test, pred_lstm)
    ]

    rmse_scores = [
        np.sqrt(mean_squared_error(y_test, pred_rf)),
        np.sqrt(mean_squared_error(y_test, pred_cnn)),
        np.sqrt(mean_squared_error(y_test, pred_lstm))
    ]

    mae_scores = [
        mean_absolute_error(y_test, pred_rf),
        mean_absolute_error(y_test, pred_cnn),
        mean_absolute_error(y_test, pred_lstm)
    ]

    mse_scores = [
        mean_squared_error(y_test, pred_rf),
        mean_squared_error(y_test, pred_cnn),
        mean_squared_error(y_test, pred_lstm)
    ]

    # 3. 2x2 Bar Chart Visualization
    plt.figure(figsize=(14, 10))

    # Plot A: R2 Score
    plt.subplot(2, 2, 1)
    sns.barplot(x=model_names, y=r2_scores, palette="viridis")
    plt.title("R2 Score (Higher is Better)")
    plt.ylim(min(min(r2_scores), 0) - 0.1, 1.0) # Adjust ylim to show negative values if present
    for i, v in enumerate(r2_scores):
        plt.text(i, v + (0.01 if v>0 else -0.05), f"{v:.3f}", ha='center', fontweight='bold')

    # Plot B: RMSE
    plt.subplot(2, 2, 2)
    sns.barplot(x=model_names, y=rmse_scores, palette="magma")
    plt.title("RMSE (Lower is Better)")
    for i, v in enumerate(rmse_scores):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    # Plot C: MAE (New)
    plt.subplot(2, 2, 3)
    sns.barplot(x=model_names, y=mae_scores, palette="plasma")
    plt.title("MAE (Lower is Better)")
    for i, v in enumerate(mae_scores):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    # Plot D: MSE (New)
    plt.subplot(2, 2, 4)
    sns.barplot(x=model_names, y=mse_scores, palette="inferno")
    plt.title("MSE (Lower is Better)")
    for i, v in enumerate(mse_scores):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 4. Print Scores
    print(f"Final R2 Scores: {dict(zip(model_names, r2_scores))}")
    print(f"Final RMSE Scores: {dict(zip(model_names, rmse_scores))}")
    print(f"Final MAE Scores: {dict(zip(model_names, mae_scores))}")
    print(f"Final MSE Scores: {dict(zip(model_names, mse_scores))}")
