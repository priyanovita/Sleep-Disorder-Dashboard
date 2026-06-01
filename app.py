# ============================================================
# app.py — Sleep Disorder Classification Dashboard
# Kelompok 17 | Python Data Science Final Project
# Streamlit Cloud ready — NO kagglehub, NO external .pkl
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder | Kel. 17",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — LIGHT THEME (putih bersih, teks hitam, selalu terbaca) ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

/* ══ RESET: paksa light mode di semua elemen Streamlit ══ */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
section[data-testid="stSidebar"] > div,
[class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Sidebar — abu-abu sangat muda */
[data-testid="stSidebar"] {
    background-color: #f3f4f6 !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #111827 !important;
    background-color: transparent !important;
}

/* Header stripe */
[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* Semua teks & heading */
h1, h2, h3, h4, h5, h6 { color: #111827 !important; }
p, li, span, label, div { color: #111827 !important; }

/* Input, textarea, select */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] [role="listbox"],
[data-baseweb="select"] [data-value] {
    background-color: #f9fafb !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
}
/* Number input */
[data-testid="stNumberInput"] input { color: #111827 !important; }

/* Slider & selectbox label */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    color: #374151 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] * {
    color: #111827 !important;
    background-color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #f3f4f6 !important;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7280 !important;
    padding: 8px 20px;
    border: none !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* ── KPI section ── */
.kpi-section {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 20px;
}
.kpi-title {
    font-size: 1.25rem; font-weight: 800;
    color: #1e3a8a; margin-bottom: 20px;
}
.kpi-grid { display: flex; gap: 0; flex-wrap: wrap; }
.kpi-item {
    flex: 1; min-width: 130px;
    border-right: 1px solid #bfdbfe;
    padding: 0 28px;
}
.kpi-item:first-child { padding-left: 0; }
.kpi-item:last-child  { border-right: none; }
.kpi-label { font-size: 0.75rem; color: #6b7280; margin-bottom: 6px; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #111827; line-height: 1.1; }
.kpi-unit  { font-size: 0.95rem; color: #6b7280; margin-left: 4px; }

/* ── Project Info card ── */
.proj-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.proj-card h4 { color: #1d4ed8; margin: 0 0 12px 0; font-size: 1rem; }
.proj-card p  { color: #4b5563; margin: 4px 0; font-size: 0.85rem; line-height: 1.6; }
.proj-card .highlight { color: #111827; font-weight: 700; }

/* ── Timeline cards ── */
.tl-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    border-top: 3px solid #16a34a;
}
.tl-card.row2 { border-top-color: #2563eb; }
.tl-num  { font-size: 0.7rem; font-weight: 700; color: #16a34a; letter-spacing: 0.08em; margin-bottom: 6px; }
.tl-card.row2 .tl-num { color: #2563eb; }
.tl-name { font-size: 0.9rem; font-weight: 700; color: #111827; margin-bottom: 6px; }
.tl-desc { font-size: 0.75rem; color: #6b7280; line-height: 1.5; }

/* ── Prediction box ── */
.pred-box {
    border-radius: 14px; padding: 22px; text-align: center;
    font-size: 1.4rem; font-weight: 800; margin: 14px 0;
}
.pred-normal   { background: #dcfce7; color: #15803d; border: 2px solid #22c55e; }
.pred-insomnia { background: #f3e8ff; color: #7e22ce; border: 2px solid #a855f7; }
.pred-apnea    { background: #fee2e2; color: #b91c1c; border: 2px solid #ef4444; }

/* ── BMI pill ── */
.bmi-pill {
    display: inline-block; padding: 5px 16px;
    border-radius: 999px; font-weight: 700; font-size: 0.85rem;
}
.bmi-normal     { background: #dcfce7; color: #15803d; border: 1px solid #22c55e; }
.bmi-overweight { background: #fef3c7; color: #b45309; border: 1px solid #f59e0b; }
.bmi-obese      { background: #fee2e2; color: #b91c1c; border: 1px solid #ef4444; }
.bmi-under      { background: #dbeafe; color: #1d4ed8; border: 1px solid #3b82f6; }

/* Sidebar model badge */
.sidebar-badge {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 14px 16px;
}
.sidebar-badge .badge-label { color: #d97706; font-weight: 700; font-size: 0.85rem; }
.sidebar-badge .badge-name  { color: #1d4ed8; font-weight: 800; font-size: 0.95rem; margin: 4px 0; }
.sidebar-badge .badge-acc   { color: #16a34a; font-size: 0.85rem; }

/* Footer */
.footer-txt { text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 28px 0 8px; }

/* Divider */
hr { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────
def calc_bmi(weight_kg, height_cm):
    if height_cm <= 0:
        return 0.0, "Normal"
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25.0:
        cat = "Normal"
    elif bmi < 30.0:
        cat = "Overweight"
    else:
        cat = "Obese"
    return round(bmi, 1), cat

def map_bmi_to_dataset(bmi_cat, le_classes):
    """Map calculated BMI category to nearest label in dataset's LabelEncoder."""
    # Dataset has: Normal, Normal Weight, Overweight, Obese
    priority = {
        "Normal":      ["Normal", "Normal Weight"],
        "Underweight": ["Normal Weight", "Normal"],
        "Overweight":  ["Overweight"],
        "Obese":       ["Obese"],
    }
    for cand in priority.get(bmi_cat, ["Normal"]):
        if cand in le_classes:
            return cand
    return le_classes[0]

# ─── Load Dataset ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    paths = [
        "Sleep_health_and_lifestyle_dataset.csv",
        "data/Sleep_health_and_lifestyle_dataset.csv",
        "dataset/Sleep_health_and_lifestyle_dataset.csv",
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        st.warning("⚠️ CSV tidak ditemukan — menggunakan data sintetis untuk demo.")
        np.random.seed(42)
        n = 374
        df = pd.DataFrame({
            "Person ID": range(1, n+1),
            "Gender":    np.random.choice(["Male","Female"], n),
            "Age":       np.random.randint(25, 60, n),
            "Occupation": np.random.choice([
                "Nurse","Doctor","Engineer","Teacher","Accountant",
                "Lawyer","Salesperson","Software Engineer","Scientist","Manager"], n),
            "Sleep Duration":          np.round(np.random.uniform(5.8, 8.5, n), 1),
            "Quality of Sleep":        np.random.randint(4, 10, n),
            "Physical Activity Level": np.random.randint(30, 90, n),
            "Stress Level":            np.random.randint(3, 9, n),
            "BMI Category":  np.random.choice(["Normal","Normal Weight","Overweight","Obese"], n),
            "Blood Pressure": [f"{np.random.randint(115,140)}/{np.random.randint(75,95)}" for _ in range(n)],
            "Heart Rate":  np.random.randint(65, 86, n),
            "Daily Steps": np.random.randint(3000, 10001, n),
            "Sleep Disorder": np.random.choice(
                ["Normal","Insomnia","Sleep Apnea"], n, p=[0.585,0.206,0.209]),
        })

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # KEY FIX: NaN = tidak ada gangguan = Normal
    df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("Normal")

    if "Blood_Pressure" in df.columns:
        df[["Systolic_BP","Diastolic_BP"]] = (
            df["Blood_Pressure"].str.split("/", expand=True).astype(int))
        df.drop(columns=["Blood_Pressure"], inplace=True)
    if "Person_ID" in df.columns:
        df.drop(columns=["Person_ID"], inplace=True)
    return df


# ─── Train Models ─────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df = _df.copy()

    le_gender     = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi        = LabelEncoder()
    le_target     = LabelEncoder()

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
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_train_s, y_train)
    else:
        X_tr, y_tr = X_train_s, y_train

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    for name, mdl in models_dict.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_test_s)
        results[name] = {
            "model":  mdl,
            "acc":    accuracy_score(y_test, y_pred),
            "f1":     f1_score(y_test, y_pred, average="weighted"),
            "y_pred": y_pred,
            "y_test": y_test,
        }

    best_name = max(results, key=lambda k: results[k]["acc"])

    return {
        "models":        results,
        "best_name":     best_name,
        "best_model":    results[best_name]["model"],
        "scaler":        scaler,
        "le_target":     le_target,
        "le_gender":     le_gender,
        "le_occupation": le_occupation,
        "le_bmi":        le_bmi,
        "feature_cols":  feature_cols,
        "df_encoded":    df,
    }


# ─── Bootstrap ────────────────────────────────────────────────
df = load_and_prepare_data()
with st.spinner("🔄 Melatih model ML... (hanya sekali saat pertama load)"):
    artifacts = train_models(df)

model      = artifacts["best_model"]
scaler     = artifacts["scaler"]
le_tgt     = artifacts["le_target"]
le_gen     = artifacts["le_gender"]
le_occ     = artifacts["le_occupation"]
le_bmi_enc = artifacts["le_bmi"]
best_name  = artifacts["best_name"]
best_acc   = artifacts["models"][best_name]["acc"]


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 8px 0;">
      <div style="font-size:1.3rem;font-weight:800;color:#111827;line-height:1.2;">
        🌙 Klasifikasi Gangguan Tidur
      </div>
      <div style="font-size:0.78rem;color:#6b7280;margin-top:4px;line-height:1.4;">
        Machine Learning Berdasarkan Data Kesehatan & Gaya Hidup
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("📌 Navigasi", [
        "🏠 Beranda",
        "📊 EDA & Visualisasi",
        "🤖 Prediksi Gangguan Tidur",
        "📈 Evaluasi Model",
    ])
    st.divider()
    st.markdown(f"""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px 16px;">
      <div style="color:#d97706;font-weight:700;font-size:0.85rem;">🏆 Model Terbaik:</div>
      <div style="color:#1d4ed8;font-weight:800;font-size:0.95rem;margin:4px 0;">{best_name}</div>
      <div style="color:#16a34a;font-size:0.85rem;">✅ Akurasi: {best_acc:.2%}</div>
    </div>""", unsafe_allow_html=True)
    st.caption("Dataset: Sleep Health & Lifestyle | Kaggle")


# ════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <div style="margin-bottom: 4px;">
      <h1 style="font-size:2rem;font-weight:800;color:#111827;margin:0;line-height:1.2;">
        🌙 Klasifikasi Gangguan Tidur
      </h1>
      <p style="color:#6b7280;font-size:0.95rem;margin:6px 0 0 0;">
        Machine Learning Berdasarkan Data Kesehatan &amp; Gaya Hidup
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── KPI Ringkasan ──
    vc         = df["Sleep_Disorder"].value_counts()
    n_normal   = int(vc.get("Normal", 0))
    n_insomnia = int(vc.get("Insomnia", 0))
    n_apnea    = int(vc.get("Sleep Apnea", 0))
    total      = len(df)
    avg_sleep  = df["Sleep_Duration"].mean()
    avg_stress = df["Stress_Level"].mean()
    avg_hr     = df["Heart_Rate"].mean()

    st.markdown(f"""
    <div class="kpi-section">
      <div class="kpi-title">🎯 Ringkasan Indikator Kesehatan Utama</div>
      <div class="kpi-grid">
        <div class="kpi-item">
          <div class="kpi-label">Total Partisipan</div>
          <div class="kpi-value">{total}<span class="kpi-unit">Orang</span></div>
        </div>
        <div class="kpi-item">
          <div class="kpi-label">Rerata Durasi Tidur</div>
          <div class="kpi-value">{avg_sleep:.1f}<span class="kpi-unit">Jam</span></div>
        </div>
        <div class="kpi-item">
          <div class="kpi-label">Rerata Skala Stres</div>
          <div class="kpi-value">{avg_stress:.1f}<span class="kpi-unit">/ 10</span></div>
        </div>
        <div class="kpi-item">
          <div class="kpi-label">Rerata Detak Jantung</div>
          <div class="kpi-value">{avg_hr:.1f}<span class="kpi-unit">Bpm</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info Proyek ──
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="proj-card">
          <h4>📌 Informasi Proyek</h4>
          <p><span class="highlight">Judul:</span><br>
             Klasifikasi Gangguan Tidur Menggunakan Machine Learning<br>
             Berdasarkan Data Kesehatan dan Gaya Hidup</p>
          <p style="margin-top:10px;"><span class="highlight">Sumber Dataset:</span><br>
             Kaggle Core API — <code style="color:#1d4ed8;font-size:0.8rem;">uom190346a/sleep-health-and-lifestyle-dataset</code></p>
          <p style="margin-top:10px;"><span class="highlight">Final Project:</span> Kelompok 17</p>
          <p style="margin-top:8px;"><span class="highlight">Anggota:</span><br>
             • Priya Novitasari Dwi Yanti<br>
             • Syahreza Bantara Yudha<br>
             • Qoimam Bilqisth Az Zuhri<br>
             • Nuaimah Hafifatul Husna
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="proj-card" style="margin-top:12px;">
          <h4>🎯 Target Klasifikasi</h4>
          <p>🟢 <span class="highlight">Normal</span> — Tidak ada gangguan tidur<br>
             &nbsp;&nbsp;&nbsp;&nbsp;<em>(NaN di dataset = Normal/tidak terdiagnosis)</em></p>
          <p>🟣 <span class="highlight">Insomnia</span> — Kesulitan tidur / durasi tidak cukup</p>
          <p>🔴 <span class="highlight">Sleep Apnea</span> — Gangguan pernapasan saat tidur</p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        fig_pie = px.pie(
            values=[n_normal, n_insomnia, n_apnea],
            names=["Normal", "Insomnia", "Sleep Apnea"],
            color_discrete_sequence=["#16a34a","#7c3aed","#dc2626"],
            title="Distribusi Sleep Disorder",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#374151",
            title_font_color="#111827",
            legend=dict(font=dict(color="#374151")),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Visualisasi":
    st.title("📊 Eksplorasi Data (EDA)")
    st.divider()

    t1, t2, t3, t4 = st.tabs(["📋 Overview", "📈 Distribusi", "🔗 Korelasi", "🧩 Pola Kelompok"])

    COLORS = ["#16a34a","#7c3aed","#dc2626"]
    LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                  font_color="#374151", title_font_color="#111827")

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
            "Sleep_Duration","Quality_of_Sleep","Age",
            "Physical_Activity_Level","Stress_Level","Heart_Rate","Daily_Steps"
        ])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col_sel, color="Sleep_Disorder", barmode="overlay",
                               color_discrete_sequence=COLORS, title=f"Distribusi {col_sel}", nbins=30)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="Sleep_Disorder", y=col_sel, color="Sleep_Disorder",
                          color_discrete_sequence=COLORS, title=f"Boxplot {col_sel}")
            fig2.update_layout(**LAYOUT, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.histogram(df, x="Gender", color="Sleep_Disorder", barmode="group",
                                color_discrete_sequence=COLORS, title="Gender vs Sleep Disorder")
            fig3.update_layout(**LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.histogram(df, x="BMI_Category", color="Sleep_Disorder", barmode="group",
                                color_discrete_sequence=COLORS, title="BMI vs Sleep Disorder")
            fig4.update_layout(**LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

    with t3:
        df_enc = artifacts["df_encoded"]
        corr   = df_enc.select_dtypes(include=np.number).corr()
        fig_hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           title="Heatmap Korelasi", aspect="auto", zmin=-1, zmax=1)
        fig_hm.update_layout(**LAYOUT)
        st.plotly_chart(fig_hm, use_container_width=True)

        if "Target_enc" in corr.columns:
            top = corr["Target_enc"].drop("Target_enc").abs().sort_values(ascending=False)
            fig_top = px.bar(x=top.values, y=top.index, orientation="h",
                             color=top.values, color_continuous_scale="Blues",
                             title="Korelasi Fitur terhadap Sleep Disorder",
                             labels={"x":"Abs. Korelasi","y":""})
            fig_top.update_layout(**LAYOUT)
            st.plotly_chart(fig_top, use_container_width=True)

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig5 = px.histogram(df, x="Occupation", color="Sleep_Disorder", barmode="stack",
                                color_discrete_sequence=COLORS, title="Pekerjaan vs Sleep Disorder")
            fig5.update_layout(**LAYOUT, xaxis_tickangle=-35)
            st.plotly_chart(fig5, use_container_width=True)
        with c2:
            fig6 = px.scatter(df, x="Sleep_Duration", y="Quality_of_Sleep",
                              color="Sleep_Disorder", size="Stress_Level",
                              color_discrete_sequence=COLORS,
                              title="Durasi vs Kualitas Tidur (ukuran = Stres)")
            fig6.update_layout(**LAYOUT)
            st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDIKSI
# FIX: detak jantung pakai number_input, prediksi diperbaiki
# ════════════════════════════════════════════════════════════
elif page == "🤖 Prediksi Gangguan Tidur":
    st.title("🤖 Prediksi Gangguan Tidur")
    st.markdown(
        "<span style='color:#64748b;'>Isi data di bawah untuk mengetahui "
        "apakah kamu berisiko mengalami gangguan tidur.</span>",
        unsafe_allow_html=True)
    st.divider()

    with st.form("pred_form"):
        # ── Data Pribadi ──────────────────────────────────
        st.subheader("👤 Data Pribadi")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender     = st.selectbox("Jenis Kelamin", sorted(le_gen.classes_.tolist()))
        with c2:
            age        = st.slider("Usia (tahun)", 18, 80, 30)
        with c3:
            occupation = st.selectbox("Pekerjaan", sorted(le_occ.classes_.tolist()))

        # ── Data Tubuh ────────────────────────────────────
        st.subheader("⚖️ Data Tubuh")
        c1, c2 = st.columns(2)
        with c1:
            weight_kg = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, 0.5,
                                        format="%.1f")
        with c2:
            height_cm = st.number_input("Tinggi Badan (cm)", 100.0, 220.0, 170.0, 0.5,
                                        format="%.1f")

        # ── Pola Tidur ────────────────────────────────────
        st.subheader("😴 Pola Tidur")
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep_dur = st.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.0, 0.1)
        with c2:
            quality   = st.slider("Kualitas Tidur (1–10)", 1, 10, 7)
        with c3:
            stress    = st.slider("Tingkat Stres (1–10)", 1, 10, 5)

        # ── Aktivitas & Detak Jantung ─────────────────────
        st.subheader("💪 Aktivitas & Detak Jantung")
        c1, c2, c3 = st.columns(3)
        with c1:
            activity = st.slider("Aktivitas Fisik (menit/hari)", 0, 90, 45)
        with c2:
            steps    = st.number_input("Langkah Harian", min_value=1000,
                                       max_value=20000, value=7000, step=500)
        with c3:
            # FIX: number_input, bukan slider
            hr = st.number_input("Detak Jantung Istirahat (bpm)",
                                  min_value=40, max_value=120, value=70, step=1)

        submitted = st.form_submit_button(
            "⚡ Jalankan Komputasi Diagnosis Klasifikasi",
            use_container_width=True)

    # ── BMI preview (live, di luar form) ─────────────────
    bmi_val, bmi_cat = calc_bmi(weight_kg, height_cm)
    bmi_css = {"Normal":"bmi-normal","Underweight":"bmi-under",
               "Overweight":"bmi-overweight","Obese":"bmi-obese"}.get(bmi_cat,"bmi-normal")
    st.markdown(f"""
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;
                padding:12px 20px;margin-top:10px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.3rem;">⚖️</span>
      <span style="color:#6b7280;font-size:0.85rem;">Kalkulator BMI Otomatis:</span>
      <span style="color:#111827;font-weight:700;">Skor Anda {bmi_val}</span>
      <span class="bmi-pill {bmi_css}">— {bmi_cat}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Jalankan Prediksi ─────────────────────────────────
    if submitted:
        try:
            bmi_val, bmi_cat = calc_bmi(weight_kg, height_cm)
            bmi_ds = map_bmi_to_dataset(bmi_cat, le_bmi_enc.classes_)

            # Encode semua input
            gender_enc = le_gen.transform([gender])[0]
            occ_enc    = le_occ.transform([occupation])[0]
            bmi_enc    = le_bmi_enc.transform([bmi_ds])[0]

            # BP diestimasi dari median dataset (tidak ditampilkan ke user)
            sys_est = int(df["Systolic_BP"].median())
            dia_est = int(df["Diastolic_BP"].median())

            # Susun fitur sesuai urutan training PERSIS
            # ["Age","Sleep_Duration","Quality_of_Sleep","Physical_Activity_Level",
            #  "Stress_Level","Heart_Rate","Daily_Steps",
            #  "Systolic_BP","Diastolic_BP","Gender_enc","Occupation_enc","BMI_Category_enc"]
            X_input = np.array([[
                age,
                float(sleep_dur),
                int(quality),
                int(activity),
                int(stress),
                int(hr),
                int(steps),
                sys_est,
                dia_est,
                int(gender_enc),
                int(occ_enc),
                int(bmi_enc),
            ]], dtype=float)

            X_scaled   = scaler.transform(X_input)
            pred_enc   = model.predict(X_scaled)[0]
            pred_label = le_tgt.inverse_transform([pred_enc])[0]

            # Probabilitas
            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(X_scaled)[0]
                proba_dict = dict(zip(le_tgt.classes_, proba))
            else:
                proba_dict = {c: (1.0 if c == pred_label else 0.0)
                              for c in le_tgt.classes_}

            # ── Tampilan Hasil ────────────────────────────
            st.divider()
            st.subheader("♟ Hasil Analisis Prediktif AI")

            result_info = {
                "Normal": (
                    "pred-normal",
                    "✅ Kondisi Normal (Bebas Gangguan Tidur)",
                    "Pertahankan konsistensi jam tidur dan aktivitas fisik harian "
                    "di kisaran 6000–8000 langkah untuk menyokong fase Deep Sleep malam ini."
                ),
                "Insomnia": (
                    "pred-insomnia",
                    "⚠️ Terindikasi Insomnia",
                    "Terapkan sleep hygiene yang baik: jadwal tidur teratur, hindari kafein "
                    "setelah pukul 14.00, dan batasi paparan layar 1 jam sebelum tidur. "
                    "Konsultasi dokter jika kondisi berlanjut lebih dari 3 minggu."
                ),
                "Sleep Apnea": (
                    "pred-apnea",
                    "🚨 Terindikasi Sleep Apnea",
                    "Segera konsultasi ke dokter spesialis tidur (sleep specialist). "
                    "Sleep apnea memerlukan penanganan medis — terapi CPAP atau penyesuaian "
                    "posisi tidur dapat sangat membantu. Jangan tunda pemeriksaan."
                ),
            }
            css_cls, title_txt, advice = result_info.get(pred_label, result_info["Normal"])

            st.markdown(f'<div class="pred-box {css_cls}">{title_txt}</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns([3, 2])
            with c1:
                colors_map = {"Normal":"#16a34a","Insomnia":"#7c3aed","Sleep Apnea":"#dc2626"}
                fig_p = go.Figure(go.Bar(
                    x=list(proba_dict.values()),
                    y=list(proba_dict.keys()),
                    orientation="h",
                    marker_color=[colors_map.get(k,"#2563eb") for k in proba_dict],
                    text=[f"{v*100:.1f}%" for v in proba_dict.values()],
                    textposition="outside",
                    textfont=dict(color="#111827"),
                ))
                fig_p.update_layout(
                    title="Probabilitas per Kategori",
                    xaxis_range=[0, 1.2],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#f9fafb",
                    font_color="#374151",
                    title_font_color="#111827",
                    height=220,
                    yaxis=dict(color="#111827"),
                )
                st.plotly_chart(fig_p, use_container_width=True)

                st.markdown(f"""
                <div style="background:#fffbeb;border-left:4px solid #f59e0b;
                            border-radius:8px;padding:16px 20px;">
                  <div style="color:#b45309;font-weight:700;margin-bottom:6px;">
                    💡 Smart Recommendation:
                  </div>
                  <div style="color:#374151;font-size:0.88rem;">{advice}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("**📋 Ringkasan Input:**")
                summary = pd.DataFrame({
                    "Fitur": ["Gender","Usia","Pekerjaan","BB / TB","BMI",
                              "Durasi Tidur","Kualitas Tidur","Stres",
                              "Aktivitas","Langkah","Detak Jantung"],
                    "Nilai": [
                        gender, age, occupation,
                        f"{weight_kg:.1f} kg / {height_cm:.0f} cm",
                        f"{bmi_val} ({bmi_cat})",
                        f"{sleep_dur:.1f} jam", quality, stress,
                        f"{activity} mnt", int(steps), f"{hr} bpm",
                    ]
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.info("Pastikan semua nilai input sudah terisi dengan benar.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — EVALUASI
# FIX: Metodologi tampil sebagai timeline cards (gambar 5 style)
# ════════════════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")
    st.divider()

    res    = artifacts["models"]
    df_cmp = pd.DataFrame({
        name: {"Accuracy": v["acc"], "F1-Score (W)": v["f1"]}
        for name, v in res.items()
    }).T.sort_values("Accuracy", ascending=False)

    st.subheader("🏆 Perbandingan Semua Model")
    st.dataframe(
        df_cmp.style.highlight_max(axis=0, color="#14532d").format("{:.4f}"),
        use_container_width=True
    )

    LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                  font_color="#374151", title_font_color="#111827")
    c1, c2 = st.columns(2)
    with c1:
        fig_acc = px.bar(df_cmp.reset_index(), x="index", y="Accuracy",
                         color="Accuracy", color_continuous_scale="Viridis",
                         title="Accuracy per Model")
        fig_acc.update_layout(**LAYOUT, xaxis_tickangle=-30, showlegend=False,
                              xaxis_title="", yaxis_range=[0,1.05])
        st.plotly_chart(fig_acc, use_container_width=True)
    with c2:
        fig_f1 = px.bar(df_cmp.reset_index(), x="index", y="F1-Score (W)",
                        color="F1-Score (W)", color_continuous_scale="Magma",
                        title="F1-Score per Model")
        fig_f1.update_layout(**LAYOUT, xaxis_tickangle=-30, showlegend=False,
                             xaxis_title="", yaxis_range=[0,1.05])
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Confusion Matrix ──
    st.subheader(f"🔎 Confusion Matrix — {best_name}")
    y_pred = res[best_name]["y_pred"]
    y_test = res[best_name]["y_test"]
    labels = le_tgt.classes_
    cm     = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale="Blues",
                       labels=dict(x="Prediksi", y="Aktual"),
                       title=f"Confusion Matrix — {best_name}")
    fig_cm.update_layout(**LAYOUT)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Classification Report ──
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.divider()

    # ── Garis Waktu / Timeline Cards (style gambar 5) ──
    st.subheader("🗺️ Garis Waktu Pengembangan Sistem (10 Tahapan Kerja)")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Rangkaian modul integrasi "
        "backend dan rekayasa data secara berkala:</p>",
        unsafe_allow_html=True)

    row1 = [
        ("MODUL 01", "Relational SQL Base",  "Sinkronisasi Skema Data Relasional SQL"),
        ("MODUL 02", "Data Ingestion",       "Pengumpulan & Pemuatan Data API Kaggle"),
        ("MODUL 03", "Exploratory Data",     "Analisis Data Eksplorasi (EDA) & Korelasi"),
        ("MODUL 04", "Data Cleansing",       "Pengolahan, Imputasi & Pembersihan Data"),
        ("MODUL 05", "Feature Engineering",  "Rekayasa Fitur & Label Encoding Sklearn"),
    ]
    row2 = [
        ("MODUL 06", "Data Visualization",   "Visualisasi Data Interaktif via Plotly"),
        ("MODUL 07", "Data Splitting",       "Pembagian Data Pelatihan (80:20) & Scaling"),
        ("MODUL 08", "Model Training",       "Pelatihan Model (RF, LR, SVM, KNN, GBM)"),
        ("MODUL 09", "Model Evaluation",     "Evaluasi Metriks Akurasi & Confusion Matrix"),
        ("MODUL 10", "Cloud Deployment",     "Deployment Sistem Cerdas via Streamlit Cloud"),
    ]

    def render_row(items, row_class=""):
        cols = st.columns(5)
        for col, (num, name, desc) in zip(cols, items):
            with col:
                st.markdown(f"""
                <div class="tl-card {row_class}">
                  <div class="tl-num">{num}</div>
                  <div class="tl-name">{name}</div>
                  <div class="tl-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

    render_row(row1, "")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_row(row2, "row2")

    st.markdown("""
    <div class="footer-txt">
    🌙 Sleep Disorder Dashboard · Kelompok 17 · Dataset: Kaggle · Streamlit & Scikit-learn
    </div>""", unsafe_allow_html=True)
