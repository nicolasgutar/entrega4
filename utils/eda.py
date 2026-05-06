import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


PALETTE = "viridis"
FIG_STYLE = {"figure.facecolor": "white", "axes.facecolor": "#f8f9fa", "axes.grid": True, "grid.alpha": 0.4}


def plot_series(series: pd.Series, title: str = "", ylabel: str = "", figsize=(14, 4)) -> None:
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series.index, series.values, linewidth=0.8, color="#2196F3")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 48, title_prefix: str = "") -> None:
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(series.dropna(), lags=lags, ax=axes[0], color="#2196F3", title=f"{title_prefix} ACF")
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], color="#E91E63", title=f"{title_prefix} PACF")
        plt.tight_layout()
        plt.show()


def plot_decomposition(series: pd.Series, period: int = 24, title: str = "") -> None:
    decomp = seasonal_decompose(series.dropna(), model="additive", period=period)
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        components = [
            (decomp.observed, "Observado", "#2196F3"),
            (decomp.trend, "Tendencia", "#FF9800"),
            (decomp.seasonal, "Estacionalidad", "#4CAF50"),
            (decomp.resid, "Residual", "#9C27B0"),
        ]
        for ax, (comp, label, color) in zip(axes, components):
            ax.plot(comp, linewidth=0.8, color=color)
            ax.set_ylabel(label, fontsize=10)
        axes[0].set_title(f"Descomposición Estacional — {title}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()


def test_stationarity(series: pd.Series, variable: str = "") -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    critical_values = result[4]
    is_stationary = p_value < 0.05
    print(f"\n{'='*55}")
    print(f"  Test de Dickey-Fuller Aumentado — {variable}")
    print(f"{'='*55}")
    print(f"  Estadístico ADF : {adf_stat:.4f}")
    print(f"  p-valor          : {p_value:.6f}")
    for k, v in critical_values.items():
        print(f"  Valor Crítico {k}  : {v:.4f}")
    print(f"\n  Resultado: {'✓ ESTACIONARIA' if is_stationary else '✗ NO ESTACIONARIA'} (α = 0.05)")
    print(f"{'='*55}\n")
    return {"variable": variable, "adf": adf_stat, "p_value": p_value, "stationary": is_stationary}


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Matriz de Correlación") -> None:
    corr = df.corr()
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 9}
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.show()


def plot_hourly_seasonality(series: pd.Series, title: str = "Perfil Horario Promedio") -> None:
    df = series.to_frame(name="value")
    df["hour"] = df.index.hour
    hourly = df.groupby("hour")["value"].agg(["mean", "std"]).reset_index()
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hourly["hour"], hourly["mean"], marker="o", color="#2196F3", linewidth=2)
        ax.fill_between(
            hourly["hour"],
            hourly["mean"] - hourly["std"],
            hourly["mean"] + hourly["std"],
            alpha=0.2, color="#2196F3"
        )
        ax.set_xlabel("Hora del día", fontsize=11)
        ax.set_ylabel("Valor promedio", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.show()


def plot_weekly_seasonality(series: pd.Series, title: str = "Perfil Semanal Promedio") -> None:
    df = series.to_frame(name="value")
    df["dayofweek"] = df.index.dayofweek
    days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    weekly = df.groupby("dayofweek")["value"].agg(["mean", "std"]).reset_index()
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(weekly["dayofweek"], weekly["mean"], color="#4CAF50", alpha=0.8, edgecolor="white")
        ax.errorbar(weekly["dayofweek"], weekly["mean"], yerr=weekly["std"], fmt="none", color="gray", capsize=4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_ylabel("Valor promedio", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
