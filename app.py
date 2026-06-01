import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─── KONFIGURASI HALAMAN ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder Classification",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS KUSTOM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-size: 2.6rem; font-weight: 800; color: #1a1a2e;
    text-align: center; margin-bottom: 0.2rem;
    background: linear-gradient(135deg, #6c63ff, #3ecfcf);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 100%);
    border-radius: 14px; padding: 1.3rem 1rem; color: white;
    text-align: center; margin: 0.3rem;
}
.metric-val  { font-size: 2rem; font-weight: 800; margin: 0; }
.metric-lbl  { font-size: 0.78rem; margin: 0; opacity: 0.9; }

.result-none {
    background: #eafaf1; border-left: 6px solid #27ae60;
    padding: 1.4rem 1.6rem; border-radius: 10px; margin-top: 1rem;
}
.result-insomnia {
    background: #fdedec; border-left: 6px solid #e74c3c;
    padding: 1.4rem 1.6rem; border-radius: 10px; margin-top: 1rem;
}
.result-apnea {
    background: #fef9e7; border-left: 6px solid #f39c12;
    padding: 1.4rem 1.6rem; border-radius: 10px; margin-top: 1rem;
}
.result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.4rem; }
.result-desc  { font-size: 0.95rem; color: #555; }

.section-title {
    font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
    border-bottom: 3px solid #6c63ff; padding-bottom: 6px; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD ARTEFAK ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    return df

try:
    artifacts = load_artifacts()
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    le_target    = artifacts["le_target"]
    le_dict      = artifacts["le_dict"]
    feature_cols = artifacts["feature_cols"]
    class_names  = artifacts["class_names"]
    results_dict = artifacts["results"]       # dict { model_name: {accuracy, cv, f1, ...} }
    feat_imp_df  = artifacts["feat_imp_df"]
    best_model_name = artifacts["best_model_name"]
    df           = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Gagal memuat model: {e}\nPastikan file `model_artifacts.pkl` sudah di-upload ke repo GitHub.")
    st.stop()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Sleep Disorder App")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠  Beranda", "📊  Eksplorasi Data", "📑  Dokumentasi Proyek", "🔮  Prediksi Saya"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    best_acc = results_dict[best_model_name]["accuracy"]
    st.markdown(f"""
    **Model Aktif:** {best_model_name}  
    **Akurasi:** `{best_acc*100:.1f}%`  
    **Dataset:** Sleep Health & Lifestyle  
    **Total Data:** `{len(df)}` baris
    """)


# ════════════════════════════════════════════════════════════════════════════
# 🏠 BERANDA
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Beranda":
    st.markdown('<p class="main-header">🌙 Klasifikasi Gangguan Tidur</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning · Kesehatan & Gaya Hidup · Prediksi Interaktif</p>', unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Data", len(df), "baris"),
        ("Fitur Input", len(feature_cols), "variabel"),
        ("Kelas Target", len(class_names), "kondisi"),
        ("Akurasi Model", f"{best_acc*100:.1f}%", best_model_name),
    ]
    for col, (lbl, val, unit) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-val">{val}</p>
            <p class="metric-lbl">{lbl}<br><small>{unit}</small></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-title">Distribusi Gangguan Tidur</p>', unsafe_allow_html=True)
        dist = df["Sleep Disorder"].value_counts().reset_index()
        dist.columns = ["Kondisi", "Jumlah"]
        fig = px.pie(dist, names="Kondisi", values="Jumlah",
                     color_discrete_map={"None": "#27ae60", "Insomnia": "#e74c3c", "Sleep Apnea": "#f39c12"},
                     hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10), height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-title">Durasi Tidur per Kondisi</p>', unsafe_allow_html=True)
        fig2 = px.box(df, x="Sleep Disorder", y="Sleep Duration", color="Sleep Disorder",
                      color_discrete_map={"None": "#27ae60", "Insomnia": "#e74c3c", "Sleep Apnea": "#f39c12"})
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                           xaxis_title="", yaxis_title="Jam Tidur")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Tentang Proyek</p>', unsafe_allow_html=True)
    st.markdown("""
    Dashboard ini mengklasifikasikan **gangguan tidur** (None / Insomnia / Sleep Apnea) berdasarkan
    data kesehatan dan gaya hidup menggunakan algoritma Machine Learning terbaik hasil seleksi otomatis.

    | Kondisi | Deskripsi |
    |---|---|
    | 🟢 **None** | Tidak terdeteksi gangguan tidur |
    | 🔴 **Insomnia** | Kesulitan tidur atau kualitas tidur buruk |
    | 🟡 **Sleep Apnea** | Gangguan pernapasan berulang saat tidur |

    > Gunakan menu **🔮 Prediksi Saya** untuk mengetahui kondisi tidur Anda secara personal.
    """)


# ════════════════════════════════════════════════════════════════════════════
# 📊 EKSPLORASI DATA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Eksplorasi Data":
    st.markdown('<p class="main-header" style="font-size:2rem">📊 Eksplorasi Data</p>', unsafe_allow_html=True)

    with st.expander("🔍 Lihat Dataset (10 baris pertama)"):
        st.dataframe(df.head(10), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Distribusi", "Scatter Plot", "Korelasi"])

    with tab1:
        col1, col2 = st.columns(2)
        num_col = col1.selectbox("Variabel", [
            "Age", "Sleep Duration", "Quality of Sleep",
            "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"
        ])
        color_col = col2.selectbox("Warnai berdasarkan", ["Sleep Disorder", "Gender", "BMI Category"])
        fig = px.histogram(df, x=num_col, color=color_col, barmode="overlay",
                           color_discrete_map={"None": "#27ae60", "Insomnia": "#e74c3c", "Sleep Apnea": "#f39c12"},
                           nbins=20, opacity=0.78)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Statistik ringkas
        st.markdown(f"**Statistik: {num_col}**")
        st.dataframe(df.groupby("Sleep Disorder")[num_col].describe().round(2), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        xv = c1.selectbox("Sumbu X", ["Sleep Duration", "Age", "Stress Level",
                                       "Physical Activity Level", "Heart Rate", "Daily Steps"], index=0)
        yv = c2.selectbox("Sumbu Y", ["Quality of Sleep", "Sleep Duration", "Stress Level",
                                       "Physical Activity Level", "Heart Rate"], index=0)
        fig3 = px.scatter(df, x=xv, y=yv, color="Sleep Disorder",
                          color_discrete_map={"None": "#27ae60", "Insomnia": "#e74c3c", "Sleep Apnea": "#f39c12"},
                          symbol="Gender", opacity=0.8, size_max=8)
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         aspect="auto", zmin=-1, zmax=1)
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 📑 DOKUMENTASI PROYEK
# ════════════════════════════════════════════════════════════════════════════
elif page == "📑  Dokumentasi Proyek":
    st.markdown('<p class="main-header" style="font-size:2rem">📑 Dokumentasi Proyek</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Metodologi", "🤖 Perbandingan Model", "🌟 Feature Importance", "📚 Referensi"
    ])

    # ── Tab 1: Metodologi ──────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        ### 🎯 Tujuan Proyek
        Membangun sistem klasifikasi gangguan tidur (*Sleep Disorder*) berbasis Machine Learning
        menggunakan dataset kesehatan dan gaya hidup, dengan output berupa dashboard interaktif
        yang dapat digunakan oleh pengguna umum.

        ---
        ### 📦 Dataset
        - **Sumber:** [Kaggle — Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
        - **Ukuran:** 374 baris × 13 kolom
        - **Target:** `Sleep Disorder` → None / Insomnia / Sleep Apnea

        ---
        ### 🔄 Alur Metodologi

        | Pertemuan | Tahap | Deskripsi |
        |-----------|-------|-----------|
        | 1–2 | **Import & EDA** | Load dataset, cek missing values, distribusi, statistik deskriptif |
        | 3 | **Preprocessing** | Encoding kategorikal (LabelEncoder), split Blood Pressure, fillna |
        | 4 | **Visualisasi** | Histogram, boxplot, heatmap korelasi, crosstab |
        | 5–6 | **Modeling** | Training 6 algoritma: RF, SVM, KNN, LR, DT, GB |
        | 7 | **Evaluasi** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
        | 8 | **Tuning** | GridSearchCV pada Random Forest (5-fold CV) |
        | 9 | **Interpretasi** | Feature importance, analisis kontribusi variabel |
        | 10 | **Deployment** | Streamlit dashboard, deploy ke Streamlit Community Cloud |

        ---
        ### ⚙️ Preprocessing Detail
        - **Blood Pressure** dipecah menjadi 2 fitur: `BP_Systolic` & `BP_Diastolic`
        - **Kolom kategorikal** (Gender, Occupation, BMI Category) → LabelEncoder
        - **Target** (Sleep Disorder): None → 0, Insomnia → 1 / Sleep Apnea → 2 (tergantung urutan fit)
        - **Scaling:** StandardScaler pada semua fitur numerik
        - **Train/Test Split:** 80% training, 20% testing, stratified
        """)

    # ── Tab 2: Perbandingan Model ──────────────────────────────────────────
    with tab2:
        st.markdown("### 🤖 Perbandingan Performa Semua Model")
        st.info(f"✅ **Model terpilih untuk prediksi: {best_model_name}** — dipilih berdasarkan akurasi dan CV score tertinggi.")

        # Tabel performa
        perf_rows = []
        for name, v in results_dict.items():
            perf_rows.append({
                "Model": name,
                "Test Accuracy": f"{v['accuracy']*100:.2f}%",
                "CV Score (5-fold)": f"{v['cv_score']*100:.2f}%",
                "Precision": f"{v.get('precision', 0)*100:.2f}%",
                "Recall": f"{v.get('recall', 0)*100:.2f}%",
                "F1-Score": f"{v.get('f1', 0)*100:.2f}%",
                "Status": "✅ Digunakan" if name == best_model_name else ""
            })
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(perf_df.set_index("Model"), use_container_width=True)

        # Bar chart akurasi
        st.markdown("#### Visualisasi Akurasi vs CV Score")
        raw_perf = pd.DataFrame([
            {"Model": k, "Test Accuracy": v["accuracy"], "CV Score": v["cv_score"]}
            for k, v in results_dict.items()
        ]).sort_values("Test Accuracy", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Test Accuracy", x=raw_perf["Model"], y=raw_perf["Test Accuracy"],
                             marker_color="#6c63ff", text=(raw_perf["Test Accuracy"]*100).round(1),
                             texttemplate="%{text}%", textposition="outside"))
        fig.add_trace(go.Bar(name="CV Score (5-fold)", x=raw_perf["Model"], y=raw_perf["CV Score"],
                             marker_color="#3ecfcf", text=(raw_perf["CV Score"]*100).round(1),
                             texttemplate="%{text}%", textposition="outside"))
        fig.update_layout(barmode="group", yaxis_range=[0, 1.15], height=420,
                          yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix (jika tersimpan)
        if "confusion_matrix" in artifacts:
            st.markdown(f"#### Confusion Matrix — {best_model_name}")
            cm = artifacts["confusion_matrix"]
            fig_cm = px.imshow(cm, text_auto=True,
                               x=class_names, y=class_names,
                               color_continuous_scale="Blues",
                               labels=dict(x="Prediksi", y="Aktual"))
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        if "classification_report" in artifacts:
            st.markdown(f"#### Classification Report — {best_model_name}")
            st.text(artifacts["classification_report"])

    # ── Tab 3: Feature Importance ──────────────────────────────────────────
    with tab3:
        st.markdown("### 🌟 Feature Importance")
        st.markdown("Variabel yang paling berpengaruh dalam menentukan prediksi gangguan tidur:")

        fig_fi = px.bar(feat_imp_df.head(12), x="Importance", y="Feature",
                        orientation="h", color="Importance",
                        color_continuous_scale="Teal",
                        text=feat_imp_df["Importance"].head(12).round(3))
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(height=480, yaxis={"autorange": "reversed"},
                              coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("**Interpretasi Top-5 Fitur:**")
        st.markdown("""
        Berdasarkan hasil Random Forest setelah tuning, fitur-fitur dengan kontribusi terbesar
        dalam klasifikasi gangguan tidur adalah variabel yang berhubungan dengan kualitas tidur,
        tekanan darah, stres, BMI, dan usia. Hal ini konsisten dengan literatur medis yang
        mengaitkan faktor-faktor tersebut dengan risiko insomnia dan sleep apnea.
        """)

    # ── Tab 4: Referensi ───────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        ### 📚 Referensi & Tools

        **Dataset**
        - Lokukaluge P. Achintha Irushan. (2023). *Sleep Health and Lifestyle Dataset*. Kaggle.
          https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

        **Library yang Digunakan**
        | Library | Versi | Fungsi |
        |---------|-------|--------|
        | scikit-learn | ≥1.3 | Algoritma ML, evaluasi, preprocessing |
        | pandas | ≥2.0 | Manipulasi data |
        | numpy | ≥1.24 | Komputasi numerik |
        | streamlit | ≥1.32 | Dashboard web |
        | plotly | ≥5.18 | Visualisasi interaktif |

        **Algoritma yang Dievaluasi**
        - Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, Decision Tree

        **Deployment**
        - Streamlit Community Cloud — https://streamlit.io/cloud
        """)


# ════════════════════════════════════════════════════════════════════════════
# 🔮 PREDIKSI SAYA
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediksi Saya":
    st.markdown('<p class="main-header" style="font-size:2rem">🔮 Prediksi Gangguan Tidur</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Isi data kesehatan & gaya hidup kamu — sistem akan menganalisis risiko gangguan tidurmu</p>',
                unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Profil Diri**")
        gender     = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age        = st.slider("Usia (tahun)", 18, 80, 30)
        occupation = st.selectbox("Pekerjaan", [
            "Accountant", "Doctor", "Engineer", "Lawyer",
            "Manager", "Nurse", "Sales Representative",
            "Salesperson", "Scientist", "Software Engineer", "Teacher"
        ])
        bmi = st.selectbox("Kategori BMI", ["Normal", "Normal Weight", "Obese", "Overweight"])

    with col2:
        st.markdown("**😴 Pola Tidur**")
        sleep_dur  = st.slider("Durasi Tidur (jam/malam)", 4.0, 10.0, 7.0, 0.5)
        sleep_qual = st.slider("Kualitas Tidur (1=buruk, 10=sangat baik)", 1, 10, 7)
        phys_act   = st.slider("Aktivitas Fisik (menit/hari)", 0, 120, 45)

    with col3:
        st.markdown("**❤️ Kesehatan**")
        stress      = st.slider("Tingkat Stres (1=rendah, 10=sangat tinggi)", 1, 10, 5)
        heart_rate  = st.slider("Detak Jantung Istirahat (bpm)", 50, 110, 72)
        daily_steps = st.slider("Langkah Kaki per Hari", 1000, 20000, 7000, 500)
        bp_sys      = st.slider("Tekanan Darah Sistolik (mmHg)", 90, 160, 120)
        bp_dia      = st.slider("Tekanan Darah Diastolik (mmHg)", 60, 110, 80)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Analisis Kondisi Tidur Saya", use_container_width=True, type="primary")

    if predict_btn:
        # --- Buat DataFrame input ---
        input_raw = {
            "Gender"                 : gender,
            "Age"                    : age,
            "Occupation"             : occupation,
            "Sleep Duration"         : sleep_dur,
            "Quality of Sleep"       : sleep_qual,
            "Physical Activity Level": phys_act,
            "Stress Level"           : stress,
            "BMI Category"           : bmi,
            "Heart Rate"             : heart_rate,
            "Daily Steps"            : daily_steps,
            "BP_Systolic"            : bp_sys,
            "BP_Diastolic"           : bp_dia
        }
        input_df = pd.DataFrame([input_raw])

        # --- Encode kategorikal ---
        for col in ["Gender", "Occupation", "BMI Category"]:
            if col in le_dict and col in input_df.columns:
                le = le_dict[col]
                val = input_df[col][0]
                input_df[col] = le.transform([val]) if val in le.classes_ else [0]

        # --- Align kolom ---
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        # --- Scale & Prediksi ---
        input_scaled = scaler.transform(input_df)
        pred_class   = model.predict(input_scaled)[0]
        pred_proba   = model.predict_proba(input_scaled)[0]
        pred_label   = le_target.inverse_transform([pred_class])[0]

        # --- Tampilkan Hasil ---
        st.markdown("---")
        res_col, proba_col = st.columns([1, 1])

        with res_col:
            if pred_label == "None":
                st.markdown(f"""
                <div class="result-none">
                    <p class="result-title">🟢 Tidak Ada Gangguan Tidur</p>
                    <p class="result-desc">
                        Data kamu menunjukkan pola yang <strong>sehat</strong>.
                        Tidak terdeteksi indikasi insomnia maupun sleep apnea.
                        Pertahankan gaya hidup aktif dan pola tidur teraturmu!
                    </p>
                </div>""", unsafe_allow_html=True)
            elif pred_label == "Insomnia":
                st.markdown(f"""
                <div class="result-insomnia">
                    <p class="result-title">🔴 Terindikasi Insomnia</p>
                    <p class="result-desc">
                        Data kamu menunjukkan pola yang <strong>berkaitan dengan insomnia</strong>.
                        Perhatikan kebersihan tidur (sleep hygiene) dan pertimbangkan konsultasi
                        dengan dokter jika keluhan berlanjut lebih dari 3 minggu.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-apnea">
                    <p class="result-title">🟡 Terindikasi Sleep Apnea</p>
                    <p class="result-desc">
                        Data kamu menunjukkan pola yang <strong>berkaitan dengan sleep apnea</strong>.
                        Kondisi ini memerlukan evaluasi medis. Segera konsultasikan dengan
                        dokter spesialis tidur atau paru.
                    </p>
                </div>""", unsafe_allow_html=True)

            # Rekomendasi
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**💡 Rekomendasi untuk Kamu:**")
            recs = {
                "None": [
                    "✅ Pertahankan durasi tidur 7–9 jam per malam",
                    "✅ Lanjutkan rutinitas aktivitas fisik harian",
                    "✅ Kelola stres dengan olahraga, meditasi, atau hobi",
                    "✅ Hindari kafein dan layar gadget 1 jam sebelum tidur",
                ],
                "Insomnia": [
                    "🛏️ Tetapkan jadwal tidur dan bangun yang konsisten tiap hari",
                    "📵 Jauhkan ponsel/layar dari tempat tidur",
                    "☕ Batasi kafein — hentikan konsumsi setelah jam 14.00",
                    "🧘 Coba teknik relaksasi: pernapasan 4-7-8, meditasi, atau yoga",
                    "🌡️ Pastikan kamar gelap, sejuk (~18–22°C), dan tenang",
                    "🩺 Konsultasi dokter jika berlanjut > 3 minggu",
                ],
                "Sleep Apnea": [
                    "🩺 Segera periksa ke dokter spesialis paru/tidur",
                    "⚖️ Jaga berat badan ideal — obesitas memperburuk sleep apnea",
                    "🚭 Hindari alkohol dan rokok, terutama malam hari",
                    "🛏️ Coba tidur miring (lateral) untuk membuka saluran napas",
                    "📋 Tanyakan dokter tentang pemeriksaan polisomnografi",
                    "😷 Tanyakan kemungkinan terapi CPAP jika diindikasikan",
                ]
            }
            for r in recs.get(pred_label, []):
                st.markdown(f"- {r}")

        with proba_col:
            st.markdown("**📊 Probabilitas Hasil Analisis**")
            proba_df = pd.DataFrame({
                "Kondisi": class_names,
                "Probabilitas (%)": (pred_proba * 100).round(1)
            })
            fig_p = px.bar(proba_df, x="Kondisi", y="Probabilitas (%)",
                           color="Kondisi",
                           color_discrete_map={
                               "None": "#27ae60",
                               "Insomnia": "#e74c3c",
                               "Sleep Apnea": "#f39c12"
                           },
                           text="Probabilitas (%)")
            fig_p.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_p.update_layout(showlegend=False, yaxis_range=[0, 115], height=380,
                                xaxis_title="", yaxis_title="Probabilitas (%)")
            st.plotly_chart(fig_p, use_container_width=True)

            # Gauge chart untuk kondisi terprediksi
            conf_score = max(pred_proba) * 100
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_score,
                title={"text": "Tingkat Kepercayaan Model"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6c63ff"},
                    "steps": [
                        {"range": [0, 50],  "color": "#fdecea"},
                        {"range": [50, 75], "color": "#fef9e7"},
                        {"range": [75, 100],"color": "#eafaf1"},
                    ],
                    "threshold": {"line": {"color": "#333", "width": 3}, "value": 75}
                },
                number={"suffix": "%", "font": {"size": 36}}
            ))
            fig_g.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("""
        ---
        > ⚠️ **Disclaimer:** Hasil prediksi ini bersifat **informatif** dan bukan diagnosis medis.
        > Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis profesional.
        """)
