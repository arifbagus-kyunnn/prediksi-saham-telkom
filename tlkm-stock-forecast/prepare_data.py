"""Skrip CLI untuk membersihkan dataset mentah saham TLKM.JK menjadi CSV siap pakai.

Contoh penggunaan:
    python prepare_data.py --input "SAHAM - PT Telekomunikasi Indonesia Tbk (TLKM.JK) - Sheet1.csv" --output data/tlkm_clean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecaster import clean_stock_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Membersihkan data historis saham TLKM.JK.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path file CSV mentah saham TLKM.JK.",
    )
    parser.add_argument(
        "--output",
        default="data/tlkm_clean.csv",
        help="Path file CSV bersih keluaran (default: data/tlkm_clean.csv).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {in_path}")

    df_raw = pd.read_csv(in_path)
    df_clean, report = clean_stock_data(df_raw)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    print("=== Laporan Pembersihan Data TLKM.JK ===")
    print(f"Baris awal             : {report['rows_initial']}")
    print(f"Baris bersih           : {report['rows_clean']}")
    print(f"Baris invalid dihapus  : {report['invalid_removed']}")
    print(f"Duplikat dihapus       : {report['duplicates_removed']}")
    print(f"Volume=0 diimputasi    : {report['zero_volume_imputed']} ({', '.join(report['zero_volume_dates']) or '-'})")
    print(f"Rentang tanggal        : {report['date_start']} s.d. {report['date_end']}")
    print(f"Rentang harga Close    : Rp {report['close_min']:,.0f} - Rp {report['close_max']:,.0f}")
    print(f"Tersimpan di           : {out_path}")


if __name__ == "__main__":
    main()
