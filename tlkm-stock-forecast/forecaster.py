"""Modul inti pembersihan data, rekayasa fitur, pemodelan, evaluasi, dan proyeksi saham TLKM.JK.

Logika pada modul ini dirancang identik dengan notebook
`notebook/TLKM_Stock_Prediction_Revisi.ipynb` agar hasil eksperimen dan aplikasi
Streamlit selalu konsisten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

FEATURE_COLUMNS = [
    "Close",
    "Open",
    "High",
    "Low",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "MA_5",
    "MA_10",
    "MA_20",
    "EMA_12",
    "EMA_26",
    "MACD",
    "RSI_14",
    "Volatility_10",
    "Return_1d",
    "Return_5d",
    "HL_Range_Pct",
    "OC_Change_Pct",
    "Volume_Log",
]

AVAILABLE_MODELS = [
    "Ridge Regression",
    "Linear Regression (OLS)",
    "Holt's Linear Trend",
    "Naive Baseline",
]


def _parse_volume_series(series: pd.Series) -> pd.Series:
    """Mengonversi kolom Volume (termasuk format pemisah ribuan titik) ke numerik."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Mengonversi kolom Date dari format DD/MM/YYYY atau YYYY-MM-DD."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    if parsed.isna().sum() > len(series) * 0.5:
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, dayfirst=True, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def clean_stock_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Membersihkan dataset harga saham mentah dan menghasilkan laporan pembersihan.

    Langkah pembersihan:
    1. Normalisasi nama kolom dan validasi kolom wajib.
    2. Parsing kolom Date (DD/MM/YYYY atau YYYY-MM-DD) dan konversi numerik OHLCV.
    3. Pengurutan kronologis menaik dan penghapusan duplikat tanggal.
    4. Penghapusan baris dengan Date atau Close kosong.
    5. Validasi konsistensi OHLC (High >= Low).
    6. Deteksi & imputasi anomali Volume == 0 memakai median 5 hari sebelumnya.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")

    rows_initial = int(len(df))

    df["Date"] = _parse_date_series(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = _parse_volume_series(df["Volume"])

    invalid_date_or_close = int(df["Date"].isna().sum() + df["Close"].isna().sum())
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Adj Close"])

    df = df.sort_values("Date").reset_index(drop=True)
    duplicates_removed = int(df.duplicated(subset=["Date"], keep="last").sum())
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    ohlc_invalid = int((df["High"] < df["Low"]).sum())
    if ohlc_invalid > 0:
        df = df[df["High"] >= df["Low"]].reset_index(drop=True)

    zero_vol_mask = (df["Volume"].isna()) | (df["Volume"] <= 0)
    zero_volume_dates = df.loc[zero_vol_mask, "Date"].dt.strftime("%Y-%m-%d").tolist()
    zero_volume_count = int(zero_vol_mask.sum())

    if zero_volume_count > 0:
        df.loc[zero_vol_mask, "Volume"] = np.nan
        rolling_med = df["Volume"].rolling(window=5, min_periods=1).median()
        df["Volume"] = df["Volume"].fillna(rolling_med).fillna(df["Volume"].median())

    df["Volume"] = df["Volume"].round().astype("int64")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df[REQUIRED_COLUMNS].reset_index(drop=True)

    report: Dict[str, Any] = {
        "rows_initial": rows_initial,
        "rows_clean": int(len(df)),
        "invalid_removed": invalid_date_or_close,
        "duplicates_removed": duplicates_removed,
        "ohlc_invalid_removed": ohlc_invalid,
        "zero_volume_imputed": zero_volume_count,
        "zero_volume_dates": zero_volume_dates,
        "date_start": str(df["Date"].iloc[0]) if len(df) else "-",
        "date_end": str(df["Date"].iloc[-1]) if len(df) else "-",
        "close_min": float(df["Close"].min()) if len(df) else 0.0,
        "close_max": float(df["Close"].max()) if len(df) else 0.0,
        "close_mean": float(df["Close"].mean()) if len(df) else 0.0,
    }
    return df, report


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Menghitung Relative Strength Index (RSI) tanpa kebocoran data masa depan."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_features(df_clean: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Membentuk fitur lag dan indikator teknikal dari data saham bersih.

    Semua fitur pada baris t hanya memakai informasi hingga hari t.
    Target `Target_Close` adalah harga `Close` pada hari bursa berikutnya (t + 1).
    """
    df = df_clean.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"].astype(float)
    df["Lag_1"] = close.shift(1)
    df["Lag_2"] = close.shift(2)
    df["Lag_3"] = close.shift(3)
    df["Lag_5"] = close.shift(5)

    df["MA_5"] = close.rolling(window=5).mean()
    df["MA_10"] = close.rolling(window=10).mean()
    df["MA_20"] = close.rolling(window=20).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["RSI_14"] = _compute_rsi(close, window=14)

    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Volatility_10"] = df["Return_1d"].rolling(window=10).std()

    df["HL_Range_Pct"] = (df["High"] - df["Low"]) / close.replace(0.0, np.nan)
    df["OC_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"].replace(0.0, np.nan)
    df["Volume_Log"] = np.log1p(df["Volume"].clip(lower=0).astype(float))

    if include_target:
        df["Target_Close"] = close.shift(-1)
        df["Target_Date"] = df["Date"].shift(-1)
        df = df.dropna(subset=FEATURE_COLUMNS + ["Target_Close"]).reset_index(drop=True)
    else:
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df


@dataclass
class LinearModelArtifact:
    """Model regresi linear/Ridge terstandardisasi dengan solusi analitik."""

    model_name: str
    feature_names: List[str]
    mean_: np.ndarray
    std_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = (x - self.mean_) / self.std_
        return x_scaled @ self.coef_ + self.intercept_


@dataclass
class HoltModelArtifact:
    """Model Holt's Linear Trend (Double Exponential Smoothing)."""

    model_name: str
    level_alpha: float
    trend_beta: float
    last_level: float
    last_trend: float

    def forecast_steps(self, steps: int) -> np.ndarray:
        h = np.arange(1, steps + 1, dtype=float)
        return self.last_level + h * self.last_trend


def fit_ridge_regression(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float = 10.0, model_name: str = "Ridge Regression"
) -> LinearModelArtifact:
    """Melatih model Ridge Regression (atau OLS saat alpha=0) secara deterministik."""
    mean_ = x_train.mean(axis=0)
    std_ = x_train.std(axis=0)
    std_ = np.where(std_ < 1e-12, 1.0, std_)
    x_scaled = (x_train - mean_) / std_

    n_features = x_scaled.shape[1]
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean

    # Solusi tertutup (X^T X + alpha I) w = X^T y
    gram = x_scaled.T @ x_scaled
    reg = alpha * np.eye(n_features)
    if alpha == 0.0:
        # Tambah jitter mikro agar stabil secara numerik pada matriks kolinear
        reg = 1e-6 * np.eye(n_features)
    coef_ = np.linalg.solve(gram + reg, x_scaled.T @ y_centered)

    return LinearModelArtifact(
        model_name=model_name,
        feature_names=list(FEATURE_COLUMNS),
        mean_=mean_,
        std_=std_,
        coef_=coef_,
        intercept_=y_mean,
        alpha=float(alpha),
    )


def _holt_one_step_predictions(
    series: np.ndarray, level_alpha: float = 0.85, trend_beta: float = 0.05
) -> Tuple[np.ndarray, HoltModelArtifact]:
    """Menghasilkan prediksi 1-langkah-ke-depan Holt's Linear Trend secara rekursif."""
    n = len(series)
    if n < 2:
        raise ValueError("Minimal dibutuhkan 2 observasi untuk Holt's Linear Trend.")

    level = float(series[0])
    trend = float(series[1] - series[0])
    preds = np.zeros(n, dtype=float)
    preds[0] = level

    for t in range(1, n):
        preds[t] = level + trend
        prev_level = level
        level = level_alpha * float(series[t]) + (1.0 - level_alpha) * (level + trend)
        trend = trend_beta * (level - prev_level) + (1.0 - trend_beta) * trend

    artifact = HoltModelArtifact(
        model_name="Holt's Linear Trend",
        level_alpha=level_alpha,
        trend_beta=trend_beta,
        last_level=level,
        last_trend=trend,
    )
    return preds, artifact


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray,
) -> Dict[str, float]:
    """Menghitung MAE, RMSE, MAPE (%), R2, dan Directional Accuracy (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev_close = np.asarray(prev_close, dtype=float)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / np.where(y_true == 0.0, np.nan, y_true))) * 100.0)

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    actual_dir = np.sign(y_true - prev_close)
    pred_dir = np.sign(y_pred - prev_close)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
        "Directional Accuracy (%)": round(dir_acc, 2),
    }


def evaluate_models(
    df_clean: pd.DataFrame,
    train_ratio: float = 0.8,
    ridge_alpha: float = 10.0,
    holt_alpha: float = 0.85,
    holt_beta: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Mengevaluasi seluruh model pada pembagian kronologis Train/Test.

    Mengembalikan:
    - summary_df: tabel perbandingan metrik semua model beserta perbandingan terhadap Naive Baseline.
    - predictions_map: dictionary nama model -> DataFrame berisi Date, Actual, Predicted, Prev_Close, Residual.
    """
    if not 0.5 <= train_ratio <= 0.95:
        raise ValueError("train_ratio harus berada di rentang 0.50 hingga 0.95.")

    feat_df = build_features(df_clean, include_target=True)
    if len(feat_df) < 40:
        raise ValueError("Data terlalu sedikit setelah pembentukan fitur (minimal 40 baris).")

    split_idx = int(len(feat_df) * train_ratio)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:].reset_index(drop=True)

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["Target_Close"].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["Target_Close"].to_numpy(dtype=float)
    prev_close_test = test_df["Close"].to_numpy(dtype=float)
    target_dates = test_df["Target_Date"]

    # 1. Ridge Regression
    ridge_model = fit_ridge_regression(x_train, y_train, alpha=ridge_alpha, model_name="Ridge Regression")
    pred_ridge = ridge_model.predict(x_test)

    # 2. Linear Regression (OLS)
    ols_model = fit_ridge_regression(x_train, y_train, alpha=0.0, model_name="Linear Regression (OLS)")
    pred_ols = ols_model.predict(x_test)

    # 3. Holt's Linear Trend (one-step rolling forecast across full target series aligned to test)
    full_close = np.concatenate([train_df["Close"].to_numpy(dtype=float), test_df["Close"].to_numpy(dtype=float), [y_test[-1]]])
    holt_all_preds, _ = _holt_one_step_predictions(full_close, level_alpha=holt_alpha, trend_beta=holt_beta)
    # Prediksi untuk Target_Close[i] berada pada indeks split_idx + 1 + i
    pred_holt = holt_all_preds[split_idx + 1 : split_idx + 1 + len(test_df)]

    # 4. Naive Baseline (Persistence: y_hat_{t+1} = Close_t)
    pred_naive = prev_close_test.copy()

    model_preds = {
        "Ridge Regression": pred_ridge,
        "Linear Regression (OLS)": pred_ols,
        "Holt's Linear Trend": pred_holt,
        "Naive Baseline": pred_naive,
    }

    rows = []
    predictions_map: Dict[str, pd.DataFrame] = {}

    naive_metrics = compute_metrics(y_test, pred_naive, prev_close_test)
    naive_rmse = naive_metrics["RMSE"]

    for name, preds in model_preds.items():
        m = compute_metrics(y_test, preds, prev_close_test)
        vs_naive = ((naive_rmse - m["RMSE"]) / naive_rmse * 100.0) if naive_rmse > 0 else 0.0
        rows.append(
            {
                "Model": name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MAPE (%)": m["MAPE (%)"],
                "R2": m["R2"],
                "Directional Accuracy (%)": m["Directional Accuracy (%)"],
                "Perbaikan RMSE vs Naive (%)": round(vs_naive, 2),
                "Jumlah Data Latih": int(len(train_df)),
                "Jumlah Data Uji": int(len(test_df)),
            }
        )
        predictions_map[name] = pd.DataFrame(
            {
                "Date": target_dates,
                "Prev_Close": prev_close_test,
                "Actual": y_test,
                "Predicted": np.round(preds, 2),
                "Residual": np.round(y_test - preds, 2),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return summary_df, predictions_map


def forecast_future(
    df_clean: pd.DataFrame,
    model_name: str = "Ridge Regression",
    horizon: int = 7,
    ridge_alpha: float = 10.0,
    holt_alpha: float = 0.85,
    holt_beta: float = 0.05,
) -> pd.DataFrame:
    """Membuat proyeksi harga penutupan untuk `horizon` hari bursa ke depan secara rekursif."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model tidak dikenali: {model_name}. Pilih salah satu dari {AVAILABLE_MODELS}.")
    if horizon < 1 or horizon > 60:
        raise ValueError("Horizon prediksi harus antara 1 hingga 60 hari bursa.")

    df_work = df_clean.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = df_work.sort_values("Date").reset_index(drop=True)

    last_date = df_work["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    if model_name == "Naive Baseline":
        last_close = float(df_work["Close"].iloc[-1])
        preds = [last_close] * horizon
    elif model_name == "Holt's Linear Trend":
        _, artifact = _holt_one_step_predictions(
            df_work["Close"].to_numpy(dtype=float),
            level_alpha=holt_alpha,
            trend_beta=holt_beta,
        )
        preds = artifact.forecast_steps(horizon).tolist()
    else:
        feat_train = build_features(df_work, include_target=True)
        x_all = feat_train[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_all = feat_train["Target_Close"].to_numpy(dtype=float)
        alpha = ridge_alpha if model_name == "Ridge Regression" else 0.0
        model = fit_ridge_regression(x_all, y_all, alpha=alpha, model_name=model_name)

        # Proyeksi rekursif hari demi hari dengan memperbarui OHLCV sintetis secara konservatif
        sim_df = df_work.copy()
        recent_vol = int(round(sim_df["Volume"].tail(10).median()))
        recent_spread = float(((sim_df["High"] - sim_df["Low"]) / sim_df["Close"]).tail(10).median())
        preds = []

        for next_dt in future_dates:
            feat_latest = build_features(sim_df, include_target=False)
            x_last = feat_latest[FEATURE_COLUMNS].iloc[-1:].to_numpy(dtype=float)
            next_close = float(model.predict(x_last)[0])
            next_close = max(next_close, 1.0)
            preds.append(next_close)

            prev_close = float(sim_df["Close"].iloc[-1])
            half_band = next_close * recent_spread * 0.5
            new_row = {
                "Date": next_dt,
                "Open": prev_close,
                "High": max(prev_close, next_close) + half_band * 0.3,
                "Low": max(1.0, min(prev_close, next_close) - half_band * 0.3),
                "Close": next_close,
                "Adj Close": next_close,
                "Volume": recent_vol,
            }
            sim_df = pd.concat([sim_df, pd.DataFrame([new_row])], ignore_index=True)

    last_actual = float(df_work["Close"].iloc[-1])
    out = pd.DataFrame(
        {
            "Hari Ke": np.arange(1, horizon + 1),
            "Tanggal Proyeksi": future_dates.strftime("%Y-%m-%d"),
            "Prediksi Close (Rp)": np.round(preds, 2),
        }
    )
    out["Perubahan dari Terakhir (Rp)"] = np.round(out["Prediksi Close (Rp)"] - last_actual, 2)
    out["Perubahan (%)"] = np.round((out["Prediksi Close (Rp)"] - last_actual) / last_actual * 100.0, 2)
    return out
