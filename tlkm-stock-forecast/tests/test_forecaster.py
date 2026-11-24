"""Uji otomatis modul forecaster.py memakai data sintetis.

Catatan: Angka evaluasi dari pengujian ini berasal dari data sintetis
dan tidak boleh dilaporkan sebagai hasil eksperimen penelitian.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecaster import (
    AVAILABLE_MODELS,
    FEATURE_COLUMNS,
    build_features,
    clean_stock_data,
    evaluate_models,
    forecast_future,
)


def _make_synthetic_raw_df(n_days: int = 160) -> pd.DataFrame:
    """Membuat dataset saham sintetis dengan format DD/MM/YYYY dan pemisah ribuan titik."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    returns = rng.normal(0.0005, 0.015, size=n_days)
    close = 3500.0 * np.cumprod(1.0 + returns)
    close = np.round(close, 1)
    open_p = np.round(close * (1.0 + rng.normal(0.0, 0.005, size=n_days)), 1)
    high = np.round(np.maximum(open_p, close) + rng.uniform(10.0, 40.0, size=n_days), 1)
    low = np.round(np.minimum(open_p, close) - rng.uniform(10.0, 40.0, size=n_days), 1)
    adj_close = np.round(close * 0.92, 2)
    vols = rng.integers(40_000_000, 150_000_000, size=n_days)

    vol_str = [f"{v:,}".replace(",", ".") for v in vols]
    vol_str[25] = "0"  # Sisipkan anomali volume nol

    return pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "Adj Close": adj_close,
            "Close": close,
            "High": high,
            "Low": low,
            "Open": open_p,
            "Volume": vol_str,
        }
    )


class TestForecaster(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_df = _make_synthetic_raw_df(160)
        self.clean_df, self.report = clean_stock_data(self.raw_df)

    def test_clean_stock_data_parses_dates_and_volume(self) -> None:
        self.assertEqual(self.report["rows_initial"], 160)
        self.assertEqual(self.report["rows_clean"], 160)
        self.assertEqual(self.report["zero_volume_imputed"], 1)
        self.assertTrue((self.clean_df["Volume"] > 0).all())
        self.assertTrue(pd.to_datetime(self.clean_df["Date"]).is_monotonic_increasing)

    def test_build_features_has_no_nan_and_correct_columns(self) -> None:
        feat = build_features(self.clean_df, include_target=True)
        for col in FEATURE_COLUMNS + ["Target_Close"]:
            self.assertIn(col, feat.columns)
            self.assertFalse(feat[col].isna().any())
        # Target_Close baris 0 harus sama dengan Close baris berikutnya
        self.assertAlmostEqual(float(feat["Target_Close"].iloc[0]), float(feat["Close"].iloc[1]), places=5)

    def test_evaluate_models_returns_all_models(self) -> None:
        summary_df, preds_map = evaluate_models(self.clean_df, train_ratio=0.8, ridge_alpha=10.0)
        self.assertEqual(set(summary_df["Model"]), set(AVAILABLE_MODELS))
        for model_name in AVAILABLE_MODELS:
            self.assertIn(model_name, preds_map)
            self.assertGreater(len(preds_map[model_name]), 0)
            self.assertTrue((preds_map[model_name]["Predicted"] > 0).all())

    def test_forecast_future_all_models(self) -> None:
        for model_name in AVAILABLE_MODELS:
            fc = forecast_future(self.clean_df, model_name=model_name, horizon=5)
            self.assertEqual(len(fc), 5)
            self.assertTrue((fc["Prediksi Close (Rp)"] > 0).all())


if __name__ == "__main__":
    unittest.main()
