# ============================================================
# app.py — Sleep Disorder Classification Dashboard
# Streamlit Cloud ready — NO kagglehub, NO external .pkl needed
# Jalankan: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ─── Konfigurasi Halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
.metric-card {
    background: linear-gradient(135deg, #1e2a3a, #0f1f35);
    border: 1px solid #2a4a6e;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,150,255,0.1);
    margin-bottom: 8px;
}
.metric-value { font-size: 2.2rem; font-weight: 800; color: #4fc3f7; }
.metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
.pred-box {
    border-radius: 16px; padding: 28px; text-align: center;
    font-size: 1.6rem; font-weight: 800; margin: 16px 0;
}
.pred-normal  { background:linear-gradient(135deg,#1b5e20,#2e7d32); color:#a5d6a7; border:2px solid #4caf50; }
.pred-insomnia{ background:linear-gradient(135deg,#4a148c,#7b1fa2); color:#ce93d8; border:2px solid #ab47bc; }
.pred-apnea   { background:linear-gradient(135deg,#bf360c,#e64a19); color:#ffccbc; border:2px solid #ff7043; }
h1 { color: #4fc3f7 !important; }
h2, h3 { color: #81d4fa !important; }
.stTabs [data-baseweb="tab"] {
    background:#1e2a3a; border-radius:8px; color:#90a4ae;
    padding:8px 20px; border:1px solid #2a4a6e;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color:white !important; border-color:#4fc3f7 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load & Preprocess Dataset ───────────────────────────────
@st.cache_data
def load_and_prepare_data():
    # Cari CSV di berbagai lokasi umum
    candidate_paths = [
        "Sleep_health_and_lifestyle_dataset.csv",
        "data/Sleep_health_and_lifestyle_dataset.csv",
        "dataset/Sleep_health_and_lifestyle_dataset.csv",
    ]
    df = None
    for p in candidate_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        # Fallback: buat data sintetis agar app tetap berjalan
        st.warning("⚠️ File CSV tidak ditemukan. Menggunakan data sintetis untuk demo. "
                   "Letakkan `Sleep_health_and_lifestyle_dataset.csv` di root repo.")
        np.random.seed(42)
        n = 374
        genders = np.random.choice(["Male","Female"], n)
        ages    = np.random.randint(25, 60, n)
        occs    = np.random.choice(["Nurse","Doctor","Engineer","Teacher","Accountant",
                                     "Lawyer","Salesperson","Software Engineer","Scientist","Manager"], n)
        disorders_raw = np.random.choice(["Normal","Insomnia","Sleep Apnea"],
                                          n, p=[0.585, 0.206, 0.209])
        df = pd.DataFrame({
            "Person ID": range(1, n+1),
            "Gender": genders,
            "Age": ages,
            "Occupation": occs,
            "Sleep Duration": np.round(np.random.uniform(5.8, 8.5, n), 1),
            "Quality of Sleep": np.random.randint(4, 10, n),
            "Physical Activity Level": np.random.randint(30, 90, n),
            "Stress Level": np.random.randint(3, 9, n),
            "BMI Category": np.random.choice(["Normal","Normal Weight","Overweight","Obese"], n),
            "Blood Pressure": [f"{np.random.randint(115,140)}/{np.random.randint(75,95)}" for _ in range(n)],
            "Heart Rate": np.random.randint(65, 86, n),
            "Daily Steps": np.random.randint(3000, 10001, n),
            "Sleep Disorder": disorders_raw,
        })

    # Normalisasi kolom
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # FIX UTAMA: NaN → "Normal"
    df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("Normal")

    # Pisah Blood Pressure
    if "Blood_Pressure" in df.columns:
        df[["Systolic_BP", "Diastolic_BP"]] = (
            df["Blood_Pressure"].str.split("/", expand=True).astype(int)
        )
        df.drop(columns=["Blood_Pressure"], inplace=True)

    # Drop Person_ID
    if "Person_ID" in df.columns:
        df.drop(columns=["Person_ID"], inplace=True)

    return df

@st.cache_resource
def train_models(df):
    le_gender     = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi        = LabelEncoder()
    le_target     = LabelEncoder()

    df = df.copy()
    df["Gender_enc"]       = le_gender.fit_transform(df["Gender"])
    df["Occupation_enc"]   = le_occupation.fit_transform(df["Occupation"])
    df["BMI_Category_enc"] = le_bmi.fit_transform(df["BMI_Category"])
    df["Target_enc"]       = le_target.fit_transform(df["Sleep_Disorder"])

    feature_cols = [
        "Age", "Sleep_Duration", "Quality_of_Sleep", "Physical_Activity_Level",
        "Stress_Level", "Heart_Rate", "Daily_Steps",
        "Systolic_BP", "Diastolic_BP",
        "Gender_enc", "Occupation_enc", "BMI_Category_enc"
    ]

    X = df[feature_cols]
    y = df["Target_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_train_s, y_train)
    else:
        X_tr, y_tr = X_train_s, y_train

    models = {
        "Logistic Regression":   LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":         DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=5),
        "SVM":                   SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes":           GaussianNB(),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_test_s)
        results[name] = {
            "model":    mdl,
            "acc":      accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, average="weighted"),
            "y_pred":   y_pred,
            "y_test":   y_test,
        }

    best_name = max(results, key=lambda k: results[k]["acc"])

    return {
        "models":       results,
        "best_name":    best_name,
        "best_model":   results[best_name]["model"],
        "scaler":       scaler,
        "le_target":    le_target,
        "le_gender":    le_gender,
        "le_occupation":le_occupation,
        "le_bmi":       le_bmi,
        "feature_cols": feature_cols,
        "X_test_s":     X_test_s,
        "y_test":       y_test,
        "df_encoded":   df,
    }

# ─── Muat data & latih model ──────────────────────────────────
df = load_and_prepare_data()

with st.spinner("🔄 Melatih model ML... (hanya sekali)"):
    artifacts = train_models(df)

model       = artifacts["best_model"]
scaler      = artifacts["scaler"]
le_tgt      = artifacts["le_target"]
le_gen      = artifacts["le_gender"]
le_occ      = artifacts["le_occupation"]
le_bmi_enc  = artifacts["le_bmi"]
feat_cols   = artifacts["feature_cols"]
best_name   = artifacts["best_name"]

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Sleep Disorder")
    st.caption("Classification Dashboard")
    st.divider()
    page = st.radio("📌 Navigasi", [
        "🏠 Beranda",
        "📊 EDA & Visualisasi",
        "🤖 Prediksi Gangguan Tidur",
        "📈 Evaluasi Model"
    ])
    st.divider()
    st.info(f"🏆 **Model Aktif:**\n{best_name}\n\n"
            f"✅ Akurasi: {artifacts['models'][best_name]['acc']:.2%}")
    st.caption("Dataset: Sleep Health & Lifestyle | Kaggle")

# ════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.title("🌙 Klasifikasi Gangguan Tidur")
    st.markdown("##### Machine Learning Berdasarkan Data Kesehatan & Gaya Hidup")
    st.divider()

    vc = df["Sleep_Disorder"].value_counts()
    n_normal   = vc.get("Normal", 0)
    n_insomnia = vc.get("Insomnia", 0)
    n_apnea    = vc.get("Sleep Apnea", 0)
    total      = len(df)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total,      "Total Data"),
        (c2, n_normal,   "Normal 😴"),
        (c3, n_insomnia, "Insomnia 😵"),
        (c4, n_apnea,    "Sleep Apnea 😤"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("📖 Tentang Proyek")
        st.markdown("""
        Dashboard ini mengimplementasikan **7 algoritma Machine Learning**
        untuk mengklasifikasikan gangguan tidur berdasarkan data kesehatan & gaya hidup.

        **Target Klasifikasi:**
        - 🟢 **Normal** — Tidak ada gangguan tidur *(NaN di dataset = Normal)*
        - 🟣 **Insomnia** — Kesulitan tidur / durasi tidur tidak cukup
        - 🔴 **Sleep Apnea** — Gangguan pernapasan saat tidur

        **Metodologi (10 Pertemuan):**
        Mencakup EDA, preprocessing, SMOTE, Decision Tree, Random Forest,
        KNN, Naive Bayes, SVM, Gradient Boosting, dan evaluasi model lengkap.
        """)

    with col_r:
        fig_pie = px.pie(
            values=[n_normal, n_insomnia, n_apnea],
            names=["Normal", "Insomnia", "Sleep Apnea"],
            color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
            title="Distribusi Sleep Disorder",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#90a4ae",
            title_font_color="#81d4fa",
            legend=dict(font=dict(color="#90a4ae")),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Visualisasi":
    st.title("📊 Eksplorasi Data (EDA)")
    st.divider()

    t1, t2, t3, t4 = st.tabs(["📋 Overview", "📈 Distribusi", "🔗 Korelasi", "🧩 Pola Kelompok"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**5 Data Pertama:**")
            st.dataframe(df.head(), use_container_width=True)
        with c2:
            st.markdown("**Statistik Deskriptif:**")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    with t2:
        col_sel = st.selectbox("Pilih Fitur:", [
            "Sleep_Duration", "Quality_of_Sleep", "Age",
            "Physical_Activity_Level", "Stress_Level", "Heart_Rate", "Daily_Steps"
        ])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col_sel, color="Sleep_Disorder", barmode="overlay",
                               color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                               title=f"Distribusi {col_sel}", nbins=30)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="Sleep_Disorder", y=col_sel, color="Sleep_Disorder",
                          color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                          title=f"Boxplot {col_sel}")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#90a4ae", title_font_color="#81d4fa", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.histogram(df, x="Gender", color="Sleep_Disorder", barmode="group",
                                color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                                title="Gender vs Sleep Disorder")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.histogram(df, x="BMI_Category", color="Sleep_Disorder", barmode="group",
                                color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                                title="BMI Category vs Sleep Disorder")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig4, use_container_width=True)

    with t3:
        df_enc = artifacts["df_encoded"]
        num_df = df_enc.select_dtypes(include=np.number)
        corr   = num_df.corr()
        fig_hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           title="Heatmap Korelasi", aspect="auto", zmin=-1, zmax=1)
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#90a4ae",
                             title_font_color="#81d4fa")
        st.plotly_chart(fig_hm, use_container_width=True)

        if "Target_enc" in corr.columns:
            top = corr["Target_enc"].drop("Target_enc").abs().sort_values(ascending=False)
            fig_top = px.bar(x=top.values, y=top.index, orientation="h",
                             color=top.values, color_continuous_scale="Blues",
                             title="Korelasi Fitur terhadap Sleep Disorder",
                             labels={"x":"Korelasi Absolut","y":"Fitur"})
            fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig_top, use_container_width=True)

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig5 = px.histogram(df, x="Occupation", color="Sleep_Disorder", barmode="stack",
                                color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                                title="Pekerjaan vs Sleep Disorder")
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#90a4ae", title_font_color="#81d4fa",
                               xaxis_tickangle=-35)
            st.plotly_chart(fig5, use_container_width=True)
        with c2:
            fig6 = px.scatter(df, x="Sleep_Duration", y="Quality_of_Sleep",
                              color="Sleep_Disorder", size="Stress_Level",
                              color_discrete_sequence=["#4caf50","#ab47bc","#ff7043"],
                              title="Durasi vs Kualitas Tidur (ukuran = Stres)")
            fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDIKSI
# ════════════════════════════════════════════════════════════
elif page == "🤖 Prediksi Gangguan Tidur":
    st.title("🤖 Prediksi Gangguan Tidur")
    st.markdown("Isi data di bawah untuk mengetahui apakah kamu berisiko mengalami gangguan tidur.")
    st.divider()

    with st.form("pred_form"):
        st.subheader("👤 Data Pribadi")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Jenis Kelamin", sorted(le_gen.classes_.tolist()))
        with c2:
            age = st.slider("Usia (tahun)", 18, 80, 30)
        with c3:
            occupation = st.selectbox("Pekerjaan", sorted(le_occ.classes_.tolist()))

        st.subheader("😴 Pola Tidur")
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep_dur = st.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.0, 0.1)
        with c2:
            quality   = st.slider("Kualitas Tidur (1–10)", 1, 10, 7)
        with c3:
            stress    = st.slider("Tingkat Stres (1–10)", 1, 10, 5)

        st.subheader("💪 Aktivitas & Kesehatan")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            activity  = st.slider("Aktivitas Fisik (menit/hari)", 0, 90, 45)
        with c2:
            steps     = st.number_input("Langkah Harian", 1000, 20000, 7000, 500)
        with c3:
            hr        = st.slider("Detak Jantung (bpm)", 55, 100, 70)
        with c4:
            bmi_cat   = st.selectbox("Kategori BMI", sorted(le_bmi_enc.classes_.tolist()))

        st.subheader("🫀 Tekanan Darah")
        c1, c2 = st.columns(2)
        with c1:
            systolic  = st.number_input("Sistolik (mmHg)", 90, 180, 120)
        with c2:
            diastolic = st.number_input("Diastolik (mmHg)", 60, 120, 80)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang!", use_container_width=True)

    if submitted:
        try:
            gender_enc = le_gen.transform([gender])[0]
            occ_enc    = le_occ.transform([occupation])[0]
            bmi_enc    = le_bmi_enc.transform([bmi_cat])[0]

            X_input = np.array([[
                age, sleep_dur, quality, activity,
                stress, hr, int(steps),
                int(systolic), int(diastolic),
                gender_enc, occ_enc, bmi_enc
            ]])
            X_scaled = scaler.transform(X_input)

            pred_enc   = model.predict(X_scaled)[0]
            pred_label = le_tgt.inverse_transform([pred_enc])[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[0]
                proba_dict = dict(zip(le_tgt.classes_, proba))
            else:
                proba_dict = {c: (1.0 if c == pred_label else 0.0) for c in le_tgt.classes_}

            st.divider()
            st.subheader("🎯 Hasil Prediksi")

            info = {
                "Normal":      ("pred-normal",   "✅ Tidur Kamu Normal!",
                                "Pertahankan pola tidur sehatmu. Tidur 7–9 jam, aktif bergerak, dan kelola stres."),
                "Insomnia":    ("pred-insomnia",  "⚠️ Terindikasi Insomnia",
                                "Terapkan sleep hygiene: jadwal tidur teratur, hindari kafein malam, batasi layar. Konsultasi dokter jika berlanjut."),
                "Sleep Apnea": ("pred-apnea",     "🚨 Terindikasi Sleep Apnea",
                                "Segera konsultasi ke dokter spesialis tidur. Sleep apnea perlu penanganan medis (terapi CPAP / perubahan gaya hidup)."),
            }
            css_cls, title_txt, advice = info.get(pred_label, info["Normal"])

            st.markdown(f'<div class="pred-box {css_cls}">{title_txt}</div>',
                        unsafe_allow_html=True)
            st.info(f"💡 **Saran:** {advice}")

            c1, c2 = st.columns([2, 1])
            with c1:
                colors_map = {"Normal":"#4caf50","Insomnia":"#ab47bc","Sleep Apnea":"#ff7043"}
                fig_p = go.Figure(go.Bar(
                    x=list(proba_dict.values()),
                    y=list(proba_dict.keys()),
                    orientation="h",
                    marker_color=[colors_map.get(k,"#4fc3f7") for k in proba_dict],
                    text=[f"{v*100:.1f}%" for v in proba_dict.values()],
                    textposition="outside",
                ))
                fig_p.update_layout(
                    title="Probabilitas per Kategori",
                    xaxis_range=[0, 1.15],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#90a4ae",
                    title_font_color="#81d4fa",
                    height=220,
                )
                st.plotly_chart(fig_p, use_container_width=True)

            with c2:
                st.markdown("**📋 Ringkasan Input:**")
                summary = pd.DataFrame({
                    "Fitur": ["Gender","Usia","Pekerjaan","Durasi Tidur",
                              "Kualitas Tidur","Stres","Aktivitas","BMI",
                              "Sistolik","Diastolik"],
                    "Nilai": [gender, age, occupation, f"{sleep_dur} jam",
                              quality, stress, f"{activity} mnt", bmi_cat,
                              systolic, diastolic]
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")

# ════════════════════════════════════════════════════════════
# PAGE 4 — EVALUASI
# ════════════════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")
    st.divider()

    res = artifacts["models"]
    df_cmp = pd.DataFrame({
        name: {"Accuracy": v["acc"], "F1-Score (W)": v["f1"]}
        for name, v in res.items()
    }).T.sort_values("Accuracy", ascending=False)

    st.subheader("🏆 Perbandingan Semua Model")
    st.dataframe(
        df_cmp.style
              .highlight_max(axis=0, color="#1b5e20")
              .format("{:.4f}"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = px.bar(df_cmp.reset_index(), x="index", y="Accuracy",
                         color="Accuracy", color_continuous_scale="Viridis",
                         title="Accuracy per Model")
        fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#90a4ae", title_font_color="#81d4fa",
                              xaxis_tickangle=-30, showlegend=False,
                              xaxis_title="", yaxis_range=[0,1.05])
        st.plotly_chart(fig_acc, use_container_width=True)
    with c2:
        fig_f1 = px.bar(df_cmp.reset_index(), x="index", y="F1-Score (W)",
                        color="F1-Score (W)", color_continuous_scale="Magma",
                        title="F1-Score per Model")
        fig_f1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#90a4ae", title_font_color="#81d4fa",
                             xaxis_tickangle=-30, showlegend=False,
                             xaxis_title="", yaxis_range=[0,1.05])
        st.plotly_chart(fig_f1, use_container_width=True)

    # Confusion Matrix model terbaik
    st.subheader(f"🔎 Confusion Matrix — {best_name}")
    y_pred = res[best_name]["y_pred"]
    y_test = res[best_name]["y_test"]
    labels = le_tgt.classes_
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale="Blues",
                       labels=dict(x="Prediksi", y="Aktual"),
                       title=f"Confusion Matrix — {best_name}")
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#90a4ae",
                         title_font_color="#81d4fa")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.divider()
    st.subheader("📚 Metodologi (10 Pertemuan)")
    metodologi = {
        "Pertemuan 1 — Pendahuluan ML":    "Konsep supervised learning, klasifikasi vs regresi, dan pengenalan dataset.",
        "Pertemuan 2 — Pemahaman Data":    "EDA: distribusi, statistik deskriptif, visualisasi korelasi & pola.",
        "Pertemuan 3 — Persiapan Data":    "Fill NaN → Normal, encoding, StandardScaler, train-test split.",
        "Pertemuan 4 — Regresi":           "Logistic Regression sebagai baseline model klasifikasi.",
        "Pertemuan 5 — Decision Tree":     "Pohon keputusan dengan max_depth=5, Gini impurity.",
        "Pertemuan 6 — KNN & Naive Bayes": "K-Nearest Neighbors (k=5) dan Gaussian Naive Bayes.",
        "Pertemuan 7 — SVM":               "Support Vector Machine dengan kernel RBF, probability=True.",
        "Pertemuan 8 — Ensemble":          "Random Forest (200 trees) dan Gradient Boosting.",
        "Pertemuan 9 — Optimasi":          "SMOTE untuk imbalanced class, StratifiedKFold cross-validation.",
        "Pertemuan 10 — Evaluasi":         "Accuracy, F1-Score, Confusion Matrix, Classification Report, perbandingan model.",
    }
    for k, v in metodologi.items():
        with st.expander(f"📌 {k}"):
            st.write(v)

    st.markdown("""
    <div style="text-align:center;color:#546e7a;font-size:0.8rem;padding:20px;">
    🌙 Sleep Disorder Dashboard · Dataset: Kaggle · Built with Streamlit & Scikit-learn
    </div>""", unsafe_allow_html=True)
