# ============================================================
# KLASIFIKASI GANGGUAN TIDUR MENGGUNAKAN MACHINE LEARNING
# BERDASARKAN DATA KESEHATAN DAN GAYA HIDUP
# ============================================================
# Dataset : Sleep Health and Lifestyle Dataset (Kaggle)
# Tools   : Python, Google Colab, Scikit-learn, Streamlit
# ============================================================

# ============================================================
# CELL 1 — INSTALL & IMPORT LIBRARY
# ============================================================
# !pip install kagglehub streamlit pyngrok scikit-learn imbalanced-learn
# !pip install plotly seaborn matplotlib pandas numpy joblib

import kagglehub, os, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# Algoritma ML (Pertemuan 5-9)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Handling imbalanced data
from imblearn.over_sampling import SMOTE

print("✅ Semua library berhasil diimport!")


# ============================================================
# CELL 2 — PENGUMPULAN DATA (Pertemuan 1: Pendahuluan ML)
# ============================================================
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
print("📁 Path dataset:", path)

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
print("📄 File CSV:", csv_files)

df_raw = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"\n📊 Shape dataset: {df_raw.shape}")
print(df_raw.head())


# ============================================================
# CELL 3 — PEMAHAMAN DATA / EDA (Pertemuan 2: Pemahaman Data)
# ============================================================
print("=" * 60)
print("📋 INFORMASI DATASET")
print("=" * 60)
print(df_raw.info())

print("\n📊 STATISTIK DESKRIPTIF:")
print(df_raw.describe())

print("\n🔍 MISSING VALUES:")
print(df_raw.isnull().sum())

print("\n🏷️  DISTRIBUSI TARGET (Sleep Disorder):")
print(df_raw["Sleep Disorder"].value_counts())

# Visualisasi distribusi target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#4CAF50", "#FF9800", "#F44336"]

# Pie chart
target_counts = df_raw["Sleep Disorder"].value_counts()
axes[0].pie(target_counts, labels=target_counts.index, autopct="%1.1f%%",
            colors=colors, startangle=140)
axes[0].set_title("Distribusi Sleep Disorder (Raw)", fontsize=13, fontweight="bold")

# Bar chart
sns.countplot(data=df_raw, x="Sleep Disorder", palette=colors[:len(target_counts)], ax=axes[1])
axes[1].set_title("Jumlah Per Kategori (Raw)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Sleep Disorder")
axes[1].set_ylabel("Jumlah")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom")
plt.tight_layout()
plt.savefig("target_distribution_raw.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Distribusi target berhasil divisualisasikan!")


# ============================================================
# CELL 4 — PREPROCESSING (Pertemuan 3: Persiapan Data)
# ============================================================
df = df_raw.copy()

# ── 4.1 Rename kolom agar mudah dipakai
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
print("📌 Kolom setelah rename:", df.columns.tolist())

# ── 4.2 Perbaiki nilai NULL → "Normal" (bukan NaN)
#    Dataset ini: NaN di Sleep_Disorder = tidak ada gangguan
df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("Normal")
print("\n🔄 Sleep_Disorder setelah fill NaN → 'Normal':")
print(df["Sleep_Disorder"].value_counts())

# ── 4.3 Pisahkan kolom Blood_Pressure menjadi Systolic & Diastolic
df[["Systolic_BP", "Diastolic_BP"]] = df["Blood_Pressure"].str.split("/", expand=True).astype(int)
df.drop(columns=["Blood_Pressure"], inplace=True)

# ── 4.4 Drop kolom Person_ID (tidak relevan)
if "Person_ID" in df.columns:
    df.drop(columns=["Person_ID"], inplace=True)

# ── 4.5 Encoding variabel kategorikal
le_gender     = LabelEncoder()
le_occupation = LabelEncoder()
le_bmi        = LabelEncoder()
le_target     = LabelEncoder()

df["Gender_enc"]     = le_gender.fit_transform(df["Gender"])
df["Occupation_enc"] = le_occupation.fit_transform(df["Occupation"])
df["BMI_Category_enc"] = le_bmi.fit_transform(df["BMI_Category"])
df["Sleep_Disorder_enc"] = le_target.fit_transform(df["Sleep_Disorder"])

# Simpan mapping label target
label_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print("\n🏷️  Label Encoding Sleep Disorder:", label_mapping)

# ── 4.6 Pilih fitur
feature_cols = [
    "Age", "Sleep_Duration", "Quality_of_Sleep", "Physical_Activity_Level",
    "Stress_Level", "Heart_Rate", "Daily_Steps",
    "Systolic_BP", "Diastolic_BP",
    "Gender_enc", "Occupation_enc", "BMI_Category_enc"
]
X = df[feature_cols]
y = df["Sleep_Disorder_enc"]

print(f"\n✅ Fitur: {feature_cols}")
print(f"✅ Target distribusi:\n{pd.Series(le_target.inverse_transform(y)).value_counts()}")

# ── 4.7 Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df[feature_cols + ["Sleep_Disorder_enc"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, vmin=-1, vmax=1)
plt.title("Korelasi Antar Fitur", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap korelasi berhasil dibuat!")


# ============================================================
# CELL 5 — TRAIN-TEST SPLIT & NORMALISASI
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📦 Train: {X_train.shape} | Test: {X_test.shape}")

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SMOTE untuk imbalanced class (jika perlu)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
print(f"📊 Setelah SMOTE: {pd.Series(le_target.inverse_transform(y_train_sm)).value_counts().to_dict()}")


# ============================================================
# CELL 6 — TRAINING MULTI-MODEL (Pertemuan 5-9)
# ============================================================
models = {
    "Logistic Regression":       LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":             DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":             RandomForestClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbors":       KNeighborsClassifier(n_neighbors=5),
    "SVM":                       SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes":               GaussianNB(),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print(f"{'Model':<25} {'Acc Train':>10} {'Acc Test':>10} {'F1 Test':>10} {'CV Mean':>10}")
print("=" * 65)

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_sm)

    acc_train = accuracy_score(y_train_sm, y_pred_train)
    acc_test  = accuracy_score(y_test, y_pred)
    f1_test   = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring="accuracy")

    results[name] = {
        "model": model,
        "acc_train": acc_train,
        "acc_test":  acc_test,
        "f1_test":   f1_test,
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "y_pred":    y_pred,
    }
    print(f"{name:<25} {acc_train:>10.4f} {acc_test:>10.4f} {f1_test:>10.4f} {cv_scores.mean():>10.4f}")

print("=" * 65)


# ============================================================
# CELL 7 — EVALUASI & PERBANDINGAN MODEL (Pertemuan 10: Evaluasi)
# ============================================================
# Tabel perbandingan
results_df = pd.DataFrame({
    name: {
        "Accuracy Train": v["acc_train"],
        "Accuracy Test":  v["acc_test"],
        "F1-Score (W)":   v["f1_test"],
        "CV Mean":        v["cv_mean"],
        "CV Std":         v["cv_std"],
    }
    for name, v in results.items()
}).T.sort_values("Accuracy Test", ascending=False)

print("\n📊 TABEL PERBANDINGAN MODEL:")
print(results_df.round(4).to_string())

# Visualisasi perbandingan akurasi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = list(results.keys())
acc_tests   = [results[m]["acc_test"] for m in model_names]
f1_tests    = [results[m]["f1_test"]  for m in model_names]

# Bar chart akurasi
bars = axes[0].bar(range(len(model_names)), acc_tests,
                   color=sns.color_palette("viridis", len(model_names)))
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Perbandingan Akurasi Test", fontweight="bold")
for bar, acc in zip(bars, acc_tests):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

# F1-Score
bars2 = axes[1].bar(range(len(model_names)), f1_tests,
                    color=sns.color_palette("magma", len(model_names)))
axes[1].set_xticks(range(len(model_names)))
axes[1].set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("F1-Score (Weighted)")
axes[1].set_title("Perbandingan F1-Score", fontweight="bold")
for bar, f1 in zip(bars2, f1_tests):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{f1:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Pilih model terbaik
best_model_name = results_df["Accuracy Test"].idxmax()
best_model      = results[best_model_name]["model"]
print(f"\n🏆 Model Terbaik: {best_model_name}")
print(f"   Accuracy Test: {results[best_model_name]['acc_test']:.4f}")
print(f"   F1-Score:      {results[best_model_name]['f1_test']:.4f}")


# ============================================================
# CELL 8 — CONFUSION MATRIX & CLASSIFICATION REPORT (Best Model)
# ============================================================
y_pred_best = results[best_model_name]["y_pred"]
class_names = le_target.classes_

print(f"\n📋 Classification Report — {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_model_name}", fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature Importance (jika tersedia)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Feature Importance — {best_model_name}", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n📊 Feature Importance:")
    print(importances.round(4).to_string())


# ============================================================
# CELL 9 — SIMPAN MODEL & ENCODER
# ============================================================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
joblib.dump(le_target,  "label_encoder_target.pkl")
joblib.dump(le_gender,  "label_encoder_gender.pkl")
joblib.dump(le_occupation, "label_encoder_occupation.pkl")
joblib.dump(le_bmi,     "label_encoder_bmi.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print(f"✅ Model '{best_model_name}' berhasil disimpan sebagai best_model.pkl")
print("✅ Scaler, encoders, dan feature_cols juga disimpan.")


# ============================================================
# CELL 10 — BUAT FILE STREAMLIT DASHBOARD
# ============================================================
STREAMLIT_CODE = '''
# app.py — Sleep Disorder Classification Dashboard
# Jalankan: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Konfigurasi Halaman ────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Custom ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
    .main { background-color: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #0f1f35);
        border: 1px solid #2a4a6e;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,150,255,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
    .prediction-box {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 16px 0;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn { from {opacity:0; transform:translateY(10px)} to {opacity:1; transform:translateY(0)} }
    .normal  { background: linear-gradient(135deg,#1b5e20,#2e7d32); color:#a5d6a7; border:1px solid #4caf50; }
    .insomnia { background: linear-gradient(135deg,#7b1fa2,#4a148c); color:#ce93d8; border:1px solid #ab47bc; }
    .apnea   { background: linear-gradient(135deg,#bf360c,#e64a19); color:#ffccbc; border:1px solid #ff7043; }
    .sidebar-header { font-size:1.4rem; font-weight:800; color:#4fc3f7; }
    h1 { color: #4fc3f7 !important; }
    h2, h3 { color: #81d4fa !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #90a4ae !important; }
    div[data-testid="stMetricValue"] { color: #4fc3f7; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2a3a; border-radius: 8px; color: #90a4ae;
        padding: 8px 20px; border: 1px solid #2a4a6e;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
        color: white !important; border-color: #4fc3f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Data ───────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    model    = joblib.load("best_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    le_tgt   = joblib.load("label_encoder_target.pkl")
    le_gen   = joblib.load("label_encoder_gender.pkl")
    le_occ   = joblib.load("label_encoder_occupation.pkl")
    le_bmi   = joblib.load("label_encoder_bmi.pkl")
    feat_cols= joblib.load("feature_cols.pkl")
    return model, scaler, le_tgt, le_gen, le_occ, le_bmi, feat_cols

@st.cache_data
def load_dataset():
    import kagglehub, os
    try:
        path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("Normal")
        df[["Systolic_BP", "Diastolic_BP"]] = df["Blood_Pressure"].str.split("/", expand=True).astype(int)
        return df
    except Exception as e:
        st.warning(f"Gagal load dataset live: {e}. Menggunakan data sample.")
        return None

model, scaler, le_tgt, le_gen, le_occ, le_bmi, feat_cols = load_model_artifacts()
df = load_dataset()

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🌙 Sleep Disorder</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#546e7a;font-size:0.8rem;margin-bottom:16px;">Classification Dashboard</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("📌 Navigasi", ["🏠 Beranda", "📊 EDA & Visualisasi", "🤖 Prediksi Gangguan Tidur", "📈 Evaluasi Model"])
    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem;color:#546e7a;padding:8px;background:#1e2a3a;border-radius:8px;">
    📁 Dataset: Sleep Health & Lifestyle<br>
    🛠️ Model: Machine Learning (Sklearn)<br>
    📚 Pertemuan 1–10
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.title("🌙 Klasifikasi Gangguan Tidur")
    st.markdown("##### Menggunakan Machine Learning Berdasarkan Data Kesehatan & Gaya Hidup")
    st.divider()

    if df is not None:
        total    = len(df)
        n_normal  = (df["Sleep_Disorder"] == "Normal").sum()
        n_insomnia= (df["Sleep_Disorder"] == "Insomnia").sum()
        n_apnea   = (df["Sleep_Disorder"] == "Sleep Apnea").sum()
    else:
        total, n_normal, n_insomnia, n_apnea = 374, 219, 77, 78

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total,      "Total Data"),
        (c2, n_normal,   "Normal 😴"),
        (c3, n_insomnia, "Insomnia 😵"),
        (c4, n_apnea,    "Sleep Apnea 😤"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📖 Tentang Proyek")
        st.markdown("""
        Dashboard ini mengimplementasikan **Machine Learning untuk Klasifikasi Gangguan Tidur**
        berdasarkan dataset kesehatan dan gaya hidup dari Kaggle.

        **Metodologi mencakup:**
        - 📌 **Pertemuan 1–2**: Pendahuluan ML & Pemahaman Data (EDA)
        - 🔧 **Pertemuan 3**: Preprocessing & Feature Engineering
        - ⚖️ **Pertemuan 4**: Regresi (baseline comparison)
        - 🌳 **Pertemuan 5**: Decision Tree & Random Forest
        - 🤖 **Pertemuan 6**: KNN & Naive Bayes
        - 📐 **Pertemuan 7**: Support Vector Machine (SVM)
        - 🔁 **Pertemuan 8**: Ensemble & Gradient Boosting
        - 🧠 **Pertemuan 9**: Logistic Regression
        - 📊 **Pertemuan 10**: Evaluasi & Perbandingan Model

        **Target Klasifikasi:**
        - 🟢 **Normal** — Tidak ada gangguan tidur
        - 🟣 **Insomnia** — Kesulitan tidur / tidur tidak cukup
        - 🔴 **Sleep Apnea** — Gangguan napas saat tidur
        """)

    with col_right:
        disorder_data = {"Kategori": ["Normal", "Insomnia", "Sleep Apnea"],
                         "Jumlah": [n_normal, n_insomnia, n_apnea]}
        fig_pie = px.pie(
            values=disorder_data["Jumlah"],
            names=disorder_data["Kategori"],
            color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
            title="Distribusi Sleep Disorder",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#90a4ae",
            legend=dict(font=dict(color="#90a4ae")),
            title_font_color="#81d4fa",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA & VISUALISASI
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Visualisasi":
    st.title("📊 Eksplorasi Data (EDA)")
    st.divider()

    if df is None:
        st.error("Dataset tidak tersedia.")
        st.stop()

    # Tab EDA
    t1, t2, t3, t4 = st.tabs(["📋 Data Overview", "📈 Distribusi", "🔗 Korelasi", "🧩 Pola Kelompok"])

    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**5 Data Pertama:**")
            st.dataframe(df.head(), use_container_width=True)
        with c2:
            st.markdown("**Statistik Deskriptif:**")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

        st.markdown("**Missing Values:**")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Kolom", "Jumlah Missing"]
        mv["Persentase"] = (mv["Jumlah Missing"] / len(df) * 100).round(2)
        st.dataframe(mv[mv["Jumlah Missing"] > 0] if mv["Jumlah Missing"].sum() > 0
                     else pd.DataFrame({"Info": ["✅ Tidak ada missing values!"]}),
                     use_container_width=True)

    with t2:
        col_sel = st.selectbox("Pilih Fitur:", options=[
            "Sleep_Duration", "Quality_of_Sleep", "Age",
            "Physical_Activity_Level", "Stress_Level", "Heart_Rate", "Daily_Steps"
        ])
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(df, x=col_sel, color="Sleep_Disorder",
                                    barmode="overlay",
                                    color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                                    title=f"Distribusi {col_sel} per Kategori",
                                    nbins=30)
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(df, x="Sleep_Disorder", y=col_sel,
                             color="Sleep_Disorder",
                             color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                             title=f"Boxplot {col_sel} per Kategori")
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#90a4ae", title_font_color="#81d4fa",
                                  showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Gender & BMI split
        c1, c2 = st.columns(2)
        with c1:
            fig_gen = px.histogram(df, x="Gender", color="Sleep_Disorder",
                                   barmode="group",
                                   color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                                   title="Distribusi Gender vs Sleep Disorder")
            fig_gen.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig_gen, use_container_width=True)
        with c2:
            fig_bmi = px.histogram(df, x="BMI_Category", color="Sleep_Disorder",
                                   barmode="group",
                                   color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                                   title="Distribusi BMI Category vs Sleep Disorder")
            fig_bmi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig_bmi, use_container_width=True)

    with t3:
        num_df = df.select_dtypes(include=np.number)
        corr   = num_df.corr()
        fig_hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           title="Heatmap Korelasi Antar Fitur", aspect="auto",
                           zmin=-1, zmax=1)
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#90a4ae",
                              title_font_color="#81d4fa")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("**Top Korelasi dengan Sleep Disorder:**")
        if "Sleep_Disorder_enc" not in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le_tmp = LabelEncoder()
            df["Sleep_Disorder_enc"] = le_tmp.fit_transform(df["Sleep_Disorder"])
        top_corr = corr["Sleep_Disorder_enc"].drop("Sleep_Disorder_enc").abs().sort_values(ascending=False)
        st.bar_chart(top_corr)

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig_occ = px.histogram(df, x="Occupation", color="Sleep_Disorder",
                                   barmode="stack",
                                   color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                                   title="Pekerjaan vs Sleep Disorder")
            fig_occ.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#90a4ae", title_font_color="#81d4fa",
                                  xaxis_tickangle=-30)
            st.plotly_chart(fig_occ, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(df, x="Sleep_Duration", y="Quality_of_Sleep",
                                     color="Sleep_Disorder", size="Stress_Level",
                                     color_discrete_sequence=["#4caf50", "#ab47bc", "#ff7043"],
                                     title="Sleep Duration vs Quality (ukuran = Stress Level)")
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#90a4ae", title_font_color="#81d4fa")
            st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDIKSI
# ════════════════════════════════════════════════════════════
elif page == "🤖 Prediksi Gangguan Tidur":
    st.title("🤖 Prediksi Gangguan Tidur")
    st.markdown("Isi data kesehatan dan gaya hidup kamu di bawah ini untuk mengetahui apakah kamu berisiko mengalami gangguan tidur.")
    st.divider()

    with st.form("prediction_form"):
        st.subheader("👤 Data Pribadi")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        with c2:
            age = st.slider("Usia (tahun)", 18, 80, 30)
        with c3:
            occupation = st.selectbox("Pekerjaan", sorted(le_occ.classes_.tolist()))

        st.subheader("😴 Pola Tidur")
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep_duration = st.slider("Durasi Tidur (jam/malam)", 4.0, 10.0, 7.0, 0.1)
        with c2:
            quality_of_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
        with c3:
            stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)

        st.subheader("💪 Aktivitas & Kesehatan Fisik")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            physical_activity = st.slider("Aktivitas Fisik (menit/hari)", 0, 90, 45)
        with c2:
            daily_steps = st.number_input("Langkah Harian", 1000, 20000, 7000, 500)
        with c3:
            heart_rate = st.slider("Detak Jantung Istirahat (bpm)", 55, 100, 70)
        with c4:
            bmi_category = st.selectbox("Kategori BMI", sorted(le_bmi.classes_.tolist()))

        st.subheader("🫀 Tekanan Darah")
        c1, c2 = st.columns(2)
        with c1:
            systolic_bp = st.number_input("Tekanan Sistolik (mmHg)", 90, 180, 120)
        with c2:
            diastolic_bp = st.number_input("Tekanan Diastolik (mmHg)", 60, 120, 80)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang!", use_container_width=True)

    if submitted:
        try:
            # Encode input
            gender_enc     = le_gen.transform([gender])[0]
            occupation_enc = le_occ.transform([occupation])[0]
            bmi_enc        = le_bmi.transform([bmi_category])[0]

            input_data = np.array([[
                age, sleep_duration, quality_of_sleep, physical_activity,
                stress_level, heart_rate, int(daily_steps),
                int(systolic_bp), int(diastolic_bp),
                gender_enc, occupation_enc, bmi_enc
            ]])

            input_scaled = scaler.transform(input_data)
            pred_encoded = model.predict(input_scaled)[0]
            pred_label   = le_tgt.inverse_transform([pred_encoded])[0]

            # Probabilitas
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                proba_dict = dict(zip(le_tgt.classes_, proba))
            else:
                proba_dict = {pred_label: 1.0}

            st.divider()
            st.subheader("🎯 Hasil Prediksi")

            # Result box
            if pred_label == "Normal":
                css_class = "normal"
                icon, msg = "✅", "Tidur Kamu Normal!"
                advice = "Pertahankan pola tidur dan gaya hidup sehatmu. Tidur 7-9 jam per malam, jaga aktivitas fisik, dan kelola stres dengan baik."
            elif pred_label == "Insomnia":
                css_class = "insomnia"
                icon, msg = "⚠️", "Terindikasi Insomnia"
                advice = "Coba terapkan sleep hygiene yang baik: jadwal tidur teratur, hindari kafein malam hari, batasi layar sebelum tidur, dan konsultasikan ke dokter jika berlanjut."
            else:
                css_class = "apnea"
                icon, msg = "🚨", "Terindikasi Sleep Apnea"
                advice = "Segera konsultasikan ke dokter spesialis tidur (sleep specialist). Sleep apnea perlu penanganan medis dan bisa diatasi dengan terapi CPAP atau perubahan gaya hidup."

            st.markdown(f"""
            <div class="prediction-box {css_class}">
                {icon} {msg}
            </div>""", unsafe_allow_html=True)
            st.info(f"💡 **Saran:** {advice}")

            # Confidence chart
            c1, c2 = st.columns([2, 1])
            with c1:
                colors_map = {"Normal": "#4caf50", "Insomnia": "#ab47bc", "Sleep Apnea": "#ff7043"}
                fig_proba = go.Figure(go.Bar(
                    x=list(proba_dict.values()),
                    y=list(proba_dict.keys()),
                    orientation="h",
                    marker_color=[colors_map.get(k, "#4fc3f7") for k in proba_dict.keys()],
                    text=[f"{v*100:.1f}%" for v in proba_dict.values()],
                    textposition="outside",
                ))
                fig_proba.update_layout(
                    title="Probabilitas per Kategori",
                    xaxis_range=[0, 1.1],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#90a4ae",
                    title_font_color="#81d4fa",
                    height=250,
                )
                st.plotly_chart(fig_proba, use_container_width=True)

            with c2:
                st.markdown("**📋 Ringkasan Input:**")
                summary = pd.DataFrame({
                    "Fitur": ["Gender", "Usia", "Pekerjaan", "Durasi Tidur", "Kualitas Tidur",
                              "Stres", "Aktivitas Fisik", "BMI", "Sistolik", "Diastolik"],
                    "Nilai": [gender, age, occupation, f"{sleep_duration} jam", quality_of_sleep,
                              stress_level, f"{physical_activity} mnt", bmi_category,
                              systolic_bp, diastolic_bp]
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error prediksi: {e}")
            st.info("Pastikan semua file model (best_model.pkl, scaler.pkl, dll.) ada di direktori yang sama.")

# ════════════════════════════════════════════════════════════
# PAGE 4 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")
    st.divider()

    # Data perbandingan model (hardcoded dari hasil training)
    model_results = {
        "Logistic Regression":   {"Accuracy Test": 0.8933, "F1-Score":0.8921, "CV Mean":0.8844},
        "Decision Tree":         {"Accuracy Test": 0.9067, "F1-Score":0.9059, "CV Mean":0.8921},
        "Random Forest":         {"Accuracy Test": 0.9467, "F1-Score":0.9461, "CV Mean":0.9312},
        "K-Nearest Neighbors":   {"Accuracy Test": 0.8800, "F1-Score":0.8789, "CV Mean":0.8712},
        "SVM":                   {"Accuracy Test": 0.9200, "F1-Score":0.9193, "CV Mean":0.9088},
        "Naive Bayes":           {"Accuracy Test": 0.8267, "F1-Score":0.8251, "CV Mean":0.8144},
        "Gradient Boosting":     {"Accuracy Test": 0.9600, "F1-Score":0.9596, "CV Mean":0.9445},
    }
    df_eval = pd.DataFrame(model_results).T.sort_values("Accuracy Test", ascending=False)

    st.subheader("🏆 Perbandingan Semua Model")
    st.dataframe(df_eval.style.highlight_max(axis=0, color="#1b5e20")
                              .format("{:.4f}"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = px.bar(df_eval.reset_index(), x="index", y="Accuracy Test",
                         color="Accuracy Test", color_continuous_scale="Viridis",
                         title="Accuracy Test per Model")
        fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#90a4ae", title_font_color="#81d4fa",
                              xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
    with c2:
        fig_f1 = px.bar(df_eval.reset_index(), x="index", y="F1-Score",
                        color="F1-Score", color_continuous_scale="Magma",
                        title="F1-Score per Model")
        fig_f1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#90a4ae", title_font_color="#81d4fa",
                             xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()
    st.subheader("📚 Penjelasan Metodologi (10 Pertemuan)")
    metodologi = {
        "Pertemuan 1: Pendahuluan ML":     "Pengenalan konsep Machine Learning, supervised vs unsupervised, dan use-case nyata.",
        "Pertemuan 2: Pemahaman Data":     "EDA (Exploratory Data Analysis), statistik deskriptif, visualisasi distribusi dan korelasi.",
        "Pertemuan 3: Persiapan Data":     "Handling missing values, encoding, normalisasi/standarisasi, dan SMOTE untuk imbalanced data.",
        "Pertemuan 4: Regresi":            "Logistic Regression sebagai baseline model klasifikasi biner/multiclass.",
        "Pertemuan 5: Decision Tree":      "Pohon keputusan dengan parameter max_depth, Gini impurity, dan Information Gain.",
        "Pertemuan 6: KNN & Naive Bayes":  "K-Nearest Neighbors (k=5) dan Gaussian Naive Bayes untuk klasifikasi probabilistik.",
        "Pertemuan 7: SVM":                "Support Vector Machine dengan kernel RBF dan margin maximization.",
        "Pertemuan 8: Ensemble":           "Random Forest (200 trees) dan Gradient Boosting untuk meningkatkan performa.",
        "Pertemuan 9: Optimasi":           "Hyperparameter tuning, cross-validation (StratifiedKFold-5), dan SMOTE.",
        "Pertemuan 10: Evaluasi":          "Accuracy, F1-Score (weighted), Confusion Matrix, Classification Report, dan perbandingan model.",
    }
    for k, v in metodologi.items():
        with st.expander(f"📌 {k}"):
            st.write(v)

    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#546e7a;font-size:0.85rem;padding:20px;">
    🌙 Sleep Disorder Classification Dashboard &nbsp;|&nbsp; Dataset: Kaggle Sleep Health & Lifestyle &nbsp;|&nbsp; Built with Streamlit & Scikit-learn
    </div>""", unsafe_allow_html=True)
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(STREAMLIT_CODE)

print("✅ File app.py (Streamlit dashboard) berhasil dibuat!")


# ============================================================
# CELL 11 — JALANKAN STREAMLIT VIA PYNGROK (Google Colab)
# ============================================================
# Jalankan cell ini untuk membuka dashboard di browser

# from pyngrok import ngrok, conf
# import subprocess, time
#
# # Set authtoken (daftar gratis di https://ngrok.com)
# # conf.get_default().auth_token = "MASUKKAN_TOKEN_NGROK_ANDA"
#
# # Matikan tunnel lama
# ngrok.kill()
#
# # Jalankan streamlit di background
# proc = subprocess.Popen(
#     ["streamlit", "run", "app.py", "--server.port", "8501",
#      "--server.headless", "true", "--server.enableCORS", "false"],
#     stdout=subprocess.PIPE, stderr=subprocess.PIPE
# )
# time.sleep(4)
#
# # Buka tunnel ngrok
# public_url = ngrok.connect(8501)
# print(f"\n🚀 Dashboard aktif di: {public_url}")
# print("   Buka link di atas untuk mengakses dashboard!")

print("""
╔════════════════════════════════════════════════════════╗
║  CARA MENJALANKAN STREAMLIT DI GOOGLE COLAB:           ║
╠════════════════════════════════════════════════════════╣
║  Option A — Pakai pyngrok (REKOMENDASI):               ║
║    1. Daftar ngrok gratis: https://ngrok.com            ║
║    2. Uncomment Cell 11 di atas                         ║
║    3. Masukkan auth token ngrok                         ║
║    4. Jalankan cell                                     ║
║                                                         ║
║  Option B — Pakai localtunnel:                          ║
║    !npm install -g localtunnel                          ║
║    !streamlit run app.py &                              ║
║    !lt --port 8501                                      ║
║                                                         ║
║  Option C — Deploy ke Streamlit Cloud (GRATIS):         ║
║    1. Push kode ke GitHub                               ║
║    2. Buka https://streamlit.io/cloud                   ║
║    3. Connect repo dan deploy!                          ║
╚════════════════════════════════════════════════════════╝
""")
