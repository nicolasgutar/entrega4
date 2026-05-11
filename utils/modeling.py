import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FIG_STYLE = {"figure.facecolor": "white", "axes.facecolor": "#f8f9fa", "axes.grid": True, "grid.alpha": 0.4}


def evaluate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "",
    y_train_true: pd.Series = None,
    y_train_pred: np.ndarray = None,
) -> dict:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    mask = y_true_arr != 0
    mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)

    result = {
        "Modelo": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R²": round(r2, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    
    if y_train_true is not None and y_train_pred is not None:
        y_tt_arr = np.array(y_train_true)
        y_tp_arr = np.array(y_train_pred)
        mask_t = y_tt_arr != 0
        mape_t = np.mean(np.abs((y_tt_arr[mask_t] - y_tp_arr[mask_t]) / y_tt_arr[mask_t])) * 100
        mae_t = mean_absolute_error(y_tt_arr, y_tp_arr)
        rmse_t = np.sqrt(mean_squared_error(y_tt_arr, y_tp_arr))
        r2_t = r2_score(y_tt_arr, y_tp_arr)
        
        result.update({
            "MAE (Train)": round(mae_t, 4),
            "RMSE (Train)": round(rmse_t, 4),
            "MAPE (%) (Train)": round(mape_t, 4),
            "R² (Train)": round(r2_t, 4),
        })
        print("  -- INTRAMUESTRAL (TRAIN) --")
        print(f"  MAE    : {mae_t:.4f}")
        print(f"  RMSE   : {rmse_t:.4f}")
        print(f"  MAPE   : {mape_t:.2f}%")
        print(f"  R²     : {r2_t:.4f}")
        print("  -- EXTRAMUESTRAL (TEST) --")

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
    last_n_train: int = 50,
    y_train_pred: pd.Series | np.ndarray = None,
) -> None:
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_test.index)

    if y_train_pred is not None and not isinstance(y_train_pred, pd.Series):
        # We assume y_train_pred aligns with the end of y_train
        y_train_pred = pd.Series(y_train_pred, index=y_train.index[-len(y_train_pred):])

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot real data
        ax.plot(y_train.iloc[-last_n_train:], label="Real (train)", color="#90A4AE", linewidth=0.8)
        ax.plot(y_test, label="Real (test)", color="#2196F3", linewidth=1.2)
        
        # Plot train predictions
        if y_train_pred is not None:
            # slice to last_n_train
            train_plot = y_train_pred[y_train_pred.index >= y_train.index[-last_n_train]]
            ax.plot(train_plot, label="Pred (intramuestral)", color="#FF9800", linewidth=1.2, linestyle="-.")
            
        # Plot test predictions
        ax.plot(y_pred, label="Pred (extramuestral)", color="#FF5722", linewidth=1.2, linestyle="--")
        
        ax.axvline(x=y_test.index[0], color="gray", linestyle=":", linewidth=1.5)
        ax.set_title(f"Predicciones vs Real — {model_name}", fontsize=13, fontweight="bold")
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
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

def get_train_predictions(forecaster, y_train, exog_train=None):
    try:
        X_tr, _ = forecaster.create_train_X_y(y=y_train, exog=exog_train)
        preds = forecaster.estimator.predict(X_tr)
        return pd.Series(preds, index=y_train.index[-len(preds):])
    except Exception:
        return None
