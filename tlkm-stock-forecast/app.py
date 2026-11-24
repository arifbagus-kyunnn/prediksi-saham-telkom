"""Aplikasi Web Streamlit — Sistem Prediksi Harga Saham TLKM.JK.

Jalankan secara lokal dengan:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecaster import (
    AVAILABLE_MODELS,
    clean_stock_data,
    evaluate_models,
    forecast_future,
)

DEFAULT_DATA_PATH = Path("data/tlkm_clean.csv")

st.set_page_config(
    page_title="Prediksi Saham TLKM.JK",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes_or_path) -> tuple[pd.DataFrame, dict]:
    df_raw = pd.read_csv(file_bytes_or_path)
    return clean_stock_data(df_raw)


def main() -> None:
    st.title("📈 Sistem Prediksi Harga Saham PT Telekomunikasi Indonesia Tbk (TLKM.JK)")
    st.caption(
        "Time-Series Regression & Exponential Smoothing (Ridge Regression, OLS, Holt's Linear Trend) "
        "dengan Evaluasi Baseline Naive"
    )

    # Sidebar: Sumber data & parameter model
    st.sidebar.header("Pengaturan Data & Model")
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file CSV saham TLKM.JK (opsional)",
        type=["csv"],
        help="Jika kosong, aplikasi akan membaca data/tlkm_clean.csv.",
    )

    df_clean = None
    cleaning_report = None

    if uploaded_file is not None:
        try:
            df_clean, cleaning_report = load_and_clean(uploaded_file)
            st.sidebar.success("Dataset berhasil dimuat dari file unggahan.")
        except Exception as exc:
            st.sidebar.error(f"Gagal membaca CSV: {exc}")
            st.stop()
    elif DEFAULT_DATA_PATH.exists():
        df_clean, cleaning_report = load_and_clean(DEFAULT_DATA_PATH)
    else:
        st.warning(
            "Data belum tersedia (`data/tlkm_clean.csv` belum ditemukan). "
            "Jalankan `prepare_data.py` / notebook terlebih dahulu, atau unggah file CSV melalui sidebar."
        )
        st.stop()

    selected_model = st.sidebar.selectbox(
        "Model Prediksi",
        options=AVAILABLE_MODELS,
        index=0,
    )
    horizon = st.sidebar.slider(
        "Horizon Proyeksi (hari bursa)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
    )
    train_ratio = st.sidebar.slider(
        "Rasio Data Latih Kronologis (%)",
        min_value=60,
        max_value=90,
        value=80,
        step=5,
    ) / 100.0
    ridge_alpha = st.sidebar.select_slider(
        "Regularisasi Ridge (alpha)",
        options=[0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        value=10.0,
    )
    show_ma = st.sidebar.checkbox("Tampilkan Moving Average (MA 5 & MA 20)", value=True)
    lookback_days = st.sidebar.slider(
        "Rentang Grafik Historis (hari bursa terakhir)",
        min_value=60,
        max_value=len(df_clean),
        value=min(180, len(df_clean)),
        step=30,
    )

    tab_pred, tab_data, tab_eval = st.tabs(["Prediksi", "Data", "Evaluasi"])

    # ------------------------------------------------------------------
    # TAB 1: PREDIKSI
    # ------------------------------------------------------------------
    with tab_pred:
        forecast_df = forecast_future(
            df_clean,
            model_name=selected_model,
            horizon=horizon,
            ridge_alpha=ridge_alpha,
        )

        last_row = df_clean.iloc[-1]
        last_close = float(last_row["Close"])
        last_date = str(last_row["Date"])
        next_pred = float(forecast_df["Prediksi Close (Rp)"].iloc[0])
        next_diff = next_pred - last_close
        next_pct = (next_diff / last_close) * 100.0
        end_pred = float(forecast_df["Prediksi Close (Rp)"].iloc[-1])
        end_pct = ((end_pred - last_close) / last_close) * 100.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close Terakhir", f"Rp {last_close:,.0f}", f"Per {last_date}")
        c2.metric(
            "Prediksi Hari Bursa Berikutnya",
            f"Rp {next_pred:,.2f}",
            f"{next_diff:+.2f} ({next_pct:+.2f}%)",
        )
        c3.metric(
            f"Proyeksi Hari Ke-{horizon}",
            f"Rp {end_pred:,.2f}",
            f"{end_pct:+.2f}%",
        )
        c4.metric("Model Aktif", selected_model, f"Alpha={ridge_alpha}" if selected_model == "Ridge Regression" else "Default")

        # Grafik historis + proyeksi
        hist = df_clean.tail(lookback_days).copy()
        hist["Date_dt"] = pd.to_datetime(hist["Date"])
        fc_dates = pd.to_datetime(forecast_df["Tanggal Proyeksi"])

        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        ax.plot(hist["Date_dt"], hist["Close"], label="Harga Close Historis", color="#1E3A5F", linewidth=1.8)
        if show_ma:
            ma5 = df_clean["Close"].rolling(5).mean().tail(lookback_days)
            ma20 = df_clean["Close"].rolling(20).mean().tail(lookback_days)
            ax.plot(hist["Date_dt"], ma5, label="MA 5", color="#D97706", linewidth=1.1, alpha=0.85)
            ax.plot(hist["Date_dt"], ma20, label="MA 20", color="#059669", linewidth=1.2, alpha=0.85)

        # Sambungkan titik terakhir historis dengan titik pertama proyeksi
        bridge_dates = pd.concat([pd.Series([hist["Date_dt"].iloc[-1]]), fc_dates], ignore_index=True)
        bridge_vals = pd.concat(
            [pd.Series([last_close]), forecast_df["Prediksi Close (Rp)"]],
            ignore_index=True,
        )
        ax.plot(
            bridge_dates,
            bridge_vals,
            label=f"Proyeksi {selected_model} ({horizon} hari)",
            color="#B91C1C",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3.5,
        )
        ax.set_title(f"Tren Harga Penutupan & Proyeksi Saham TLKM.JK ({selected_model})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Harga (Rp)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(f"Tabel Proyeksi {horizon} Hari Bursa ke Depan")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Hasil Proyeksi (CSV)",
            data=csv_bytes,
            file_name=f"proyeksi_tlkm_{horizon}hari.csv",
            mime="text/csv",
        )
        st.info(
            "Disclaimer: Hasil prediksi bersifat edukatif dan analisis statistik historis, "
            "bukan merupakan rekomendasi jual atau beli saham."
        )

    # ------------------------------------------------------------------
    # TAB 2: DATA
    # ------------------------------------------------------------------
    with tab_data:
        st.subheader("Laporan Pembersihan Data")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baris Awal", f"{cleaning_report['rows_initial']:,}")
        m2.metric("Baris Bersih", f"{cleaning_report['rows_clean']:,}")
        m3.metric("Duplikat Dihapus", f"{cleaning_report['duplicates_removed']:,}")
        m4.metric("Volume=0 Diimputasi", f"{cleaning_report['zero_volume_imputed']:,}")

        st.write(
            f"**Rentang Tanggal:** `{cleaning_report['date_start']}` s.d. `{cleaning_report['date_end']}` | "
            f"**Rentang Harga Close:** Rp {cleaning_report['close_min']:,.0f} – Rp {cleaning_report['close_max']:,.0f} "
            f"(Rata-rata: Rp {cleaning_report['close_mean']:,.2f})"
        )
        if cleaning_report["zero_volume_dates"]:
            st.caption(
                "Tanggal dengan anomali Volume = 0 yang diimputasi memakai median 5 hari sebelumnya: "
                + ", ".join(cleaning_report["zero_volume_dates"])
            )

        df_plot = df_clean.copy()
        df_plot["Date_dt"] = pd.to_datetime(df_plot["Date"])
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        ax1.plot(df_plot["Date_dt"], df_plot["Close"], color="#1E3A5F", linewidth=1.4, label="Close")
        ax1.plot(df_plot["Date_dt"], df_plot["Adj Close"], color="#64748B", linewidth=1.1, linestyle="--", label="Adj Close")
        ax1.set_title("Riwayat Harga Penutupan & Volume Perdagangan TLKM.JK", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Harga (Rp)")
        ax1.legend(loc="best")
        ax1.grid(True, linestyle=":", alpha=0.5)

        ax2.bar(df_plot["Date_dt"], df_plot["Volume"] / 1e6, color="#B91C1C", alpha=0.65, width=2.0)
        ax2.set_ylabel("Volume (Juta)")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Statistik Deskriptif")
        st.dataframe(
            df_clean[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].describe().round(2),
            use_container_width=True,
        )

        st.subheader("Pratinjau 10 Baris Terbaru")
        st.dataframe(df_clean.tail(10).iloc[::-1], use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 3: EVALUASI
    # ------------------------------------------------------------------
    with tab_eval:
        st.subheader("Evaluasi Kinerja Model (Chronological Split)")
        st.write(
            f"Pembagian data dilakukan secara kronologis tanpa pengacakan "
            f"(**{int(train_ratio * 100)}% data latih** / **{int((1 - train_ratio) * 100)}% data uji**) "
            f"untuk mencegah kebocoran data masa depan (*data leakage*)."
        )

        if st.button("Hitung evaluasi", type="primary"):
            summary_df, preds_map = evaluate_models(
                df_clean,
                train_ratio=train_ratio,
                ridge_alpha=ridge_alpha,
            )
            st.session_state["eval_summary"] = summary_df
            st.session_state["eval_preds"] = preds_map

        if "eval_summary" in st.session_state:
            summary_df = st.session_state["eval_summary"]
            preds_map = st.session_state["eval_preds"]

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            model_plot = st.selectbox(
                "Pilih model untuk melihat kurva Aktual vs Prediksi pada Data Uji",
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(selected_model),
            )
            df_ev = preds_map[model_plot]

            fig3, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
            ax_p.plot(df_ev["Date"], df_ev["Actual"], label="Aktual (Close)", color="#1E3A5F", linewidth=1.6)
            ax_p.plot(df_ev["Date"], df_ev["Predicted"], label=f"Prediksi ({model_plot})", color="#B91C1C", linewidth=1.4, linestyle="--")
            ax_p.set_title(f"Aktual vs Prediksi pada Data Uji — {model_plot}", fontsize=12, fontweight="bold")
            ax_p.set_ylabel("Harga (Rp)")
            ax_p.legend(loc="best")
            ax_p.grid(True, linestyle=":", alpha=0.5)

            ax_r.axhline(0, color="#64748B", linewidth=1.0)
            ax_r.bar(df_ev["Date"], df_ev["Residual"], color="#D97706", alpha=0.75, width=1.5)
            ax_r.set_ylabel("Residual (Rp)")
            ax_r.grid(True, linestyle=":", alpha=0.5)
            ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig3.autofmt_xdate()
            st.pyplot(fig3)
            plt.close(fig3)

            st.caption(
                "Catatan pelaporan: Pada data saham harian, Naive Baseline (Close_{t+1} = Close_t) merupakan pembanding "
                "ketat karena sifat random walk pasar. Laporkan MAE, RMSE, MAPE, R², serta Directional Accuracy secara "
                "berdampingan untuk analisis yang objektif."
            )
        else:
            st.info("Klik tombol **Hitung evaluasi** untuk menghitung MAE, RMSE, MAPE, R², dan perbandingan terhadap Naive Baseline.")


if __name__ == "__main__":
    main()
