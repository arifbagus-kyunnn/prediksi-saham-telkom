"""Skrip CLI untuk mencetak hasil evaluasi model prediksi saham TLKM.JK di terminal.

Contoh penggunaan:
    python evaluate.py --data data/tlkm_clean.csv --train-ratio 0.8 --alpha 10.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecaster import clean_stock_data, evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Mengevaluasi model prediksi harga saham TLKM.JK.")
    parser.add_argument(
        "--data",
        default="data/tlkm_clean.csv",
        help="Path file CSV bersih (default: data/tlkm_clean.csv).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Rasio data latih kronologis (default: 0.8).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Parameter regularisasi Ridge Regression (default: 10.0).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {data_path}")

    df_raw = pd.read_csv(data_path)
    df_clean, _ = clean_stock_data(df_raw)
    summary_df, _ = evaluate_models(
        df_clean,
        train_ratio=args.train_ratio,
        ridge_alpha=args.alpha,
    )

    print("=== Hasil Evaluasi Model Prediksi Saham TLKM.JK (Data Asli) ===")
    print(f"Pembagian kronologis: {args.train_ratio*100:.0f}% latih / {(1-args.train_ratio)*100:.0f}% uji | Ridge alpha={args.alpha}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Catatan pelaporan:")
    print("- Pembagian data dilakukan secara kronologis (tanpa shuffle) untuk mencegah data leakage.")
    print("- Bandingkan metrik error (MAE, RMSE, MAPE) terhadap Naive Baseline (Close_{t+1} = Close_t).")
    print("- Hasil pada folder tests/ memakai data sintetis dan tidak boleh dilaporkan.")


if __name__ == "__main__":
    main()
