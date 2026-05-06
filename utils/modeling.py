import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FIG_STYLE = {"figure.facecolor": "white", "axes.facecolor": "#f8f9fa", "axes.grid": True, "grid.alpha": 0.4}


def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "") -> dict:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Avoid division by zero in MAPE
    mask = y_true_arr != 0
    mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)

    result = {"Modelo": model_name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE (%)": round(mape, 4), "R²": round(r2, 4)}
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  MAE    : {mae:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  MAPE   : {mape:.2f}%")
    print(f"  R²     : {r2:.4f}")
    print(f"{'='*50}\n")
    return result


def plot_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "",
    last_n_train: int = 500,
) -> None:
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_test.index)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(y_train.iloc[-last_n_train:], label="Entrenamiento", color="#90A4AE", linewidth=0.8)
        ax.plot(y_test, label="Real (test)", color="#2196F3", linewidth=1.2)
        ax.plot(y_pred, label="Predicción", color="#FF5722", linewidth=1.2, linestyle="--")
        ax.axvline(x=y_test.index[0], color="gray", linestyle=":", linewidth=1.5)
        ax.set_title(f"Predicciones vs Real — {model_name}", fontsize=13, fontweight="bold")
        ax.set_ylabel("NO₂ (µg/m³)", fontsize=11)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


def results_summary(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).sort_values("MAPE (%)", ascending=True).reset_index(drop=True)
    df.index += 1
    print("\n" + "="*65)
    print("  TABLA COMPARATIVA DE MODELOS")
    print("="*65)
    print(df.to_string())
    print("="*65 + "\n")
    return df
