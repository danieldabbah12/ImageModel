import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def generate_blood_test_dataset(n=100, noise_ratio=0.08, random_state=42):
    np.random.seed(random_state)
    half = n // 2

    normal = pd.DataFrame({
        "heart_rate": np.random.normal(72, 8, half),
        "body_temp": np.random.normal(36.6, 0.3, half),
        "protein_level": np.random.normal(7.0, 0.5, half),
        "white_blood_cells": np.random.normal(7000, 1200, half),
        "hemoglobin": np.random.normal(14, 1.5, half),
        "age": np.random.randint(18, 65, half),
        "gender": np.random.randint(0, 2, half),
        "test_normal": 1
    })
    abnormal = pd.DataFrame({
        "heart_rate": np.random.normal(90, 15, half),
        "body_temp": np.random.normal(37.8, 0.8, half),
        "protein_level": np.random.normal(6.0, 1.0, half),
        "white_blood_cells": np.random.normal(11000, 3000, half),
        "hemoglobin": np.random.normal(11.5, 2.0, half),
        "age": np.random.randint(18, 80, half),
        "gender": np.random.randint(0, 2, half),
        "test_normal": 0
    })
    df = pd.concat([normal, abnormal]).sample(frac=1).reset_index(drop=True)
    noise_size = int(n * noise_ratio)
    noise_idx = np.random.choice(n, noise_size, replace=False)
    df.loc[noise_idx, "test_normal"] = 1 - df.loc[noise_idx, "test_normal"]
    df["heart_rate"] = df["heart_rate"].round(0)
    df["body_temp"] = df["body_temp"].round(1)
    df["protein_level"] = df["protein_level"].round(2)
    df["white_blood_cells"] = df["white_blood_cells"].round(0)
    df["hemoglobin"] = df["hemoglobin"].round(1)
    return df

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="מדריך גרפים ב-Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS (RTL + style)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Assistant', sans-serif;
        direction: rtl;
        text-align: right;
    }

    .main { background-color: #f8fafc; }

    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #1a8fc1 100%);
        border-radius: 16px;
        padding: 40px 32px;
        margin-bottom: 32px;
        color: white;
    }
    .hero-banner h1 { font-size: 2.2rem; margin: 0 0 8px 0; }
    .hero-banner p  { font-size: 1.1rem; margin: 0; opacity: 0.9; }

    .section-header {
        background: white;
        border-right: 5px solid #2d6a9f;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 24px 0 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .section-header h2 { margin: 0; color: #1e3a5f; font-size: 1.5rem; }
    .section-header p  { margin: 4px 0 0 0; color: #555; font-size: 0.95rem; }

    .concept-box {
        background: #eaf4fb;
        border: 1px solid #aed6f1;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .concept-box h3 { color: #1a5276; margin: 0 0 8px 0; }

    .tip-box {
        background: #fef9e7;
        border: 1px solid #f9ca24;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .goal-box {
        background: #eafaf1;
        border: 1px solid #82e0aa;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .code-label {
        background: #1e3a5f;
        color: white;
        border-radius: 6px 6px 0 0;
        padding: 6px 14px;
        font-size: 0.82rem;
        display: inline-block;
        margin-bottom: -4px;
        direction: ltr;
    }

    .stCodeBlock { border-radius: 0 8px 8px 8px !important; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #2d6a9f; }
    .metric-card .label { font-size: 0.9rem; color: #666; }

    hr.divider {
        border: none;
        border-top: 2px dashed #d5e8f5;
        margin: 32px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
df = generate_blood_test_dataset(n=200)
df["test_label"] = df["test_normal"].map({1: "תקין", 0: "לא תקין"})

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 ניווט מהיר")
    page = st.radio(
        "בחר פרק:",
        [
            "🏠 מבוא ומטרה",
            "📖 נתון בדיד vs רציף",
            "📊 גרפים לנתון בדיד",
            "📈 גרפים לנתון רציף",
            "🎯 גרפים לחיזוי test_normal",
            "🔗 קורלציות בין משתנים",
        ]
    )
    st.markdown("---")
    st.markdown("**הדאטהסט שלנו:**")
    st.markdown(f"- **{len(df)}** רשומות")
    st.markdown(f"- **{df.columns.nunique()}** עמודות")
    n_normal = (df["test_normal"] == 1).sum()
    n_abnormal = (df["test_normal"] == 0).sum()
    st.markdown(f"- ✅ תקין: **{n_normal}**")
    st.markdown(f"- ❌ לא תקין: **{n_abnormal}**")

# ═══════════════════════════════════════════════════════
# PAGE 1 – מבוא
# ═══════════════════════════════════════════════════════
if page == "🏠 מבוא ומטרה":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 מדריך ויזואליזציה ב-Streamlit</h1>
        <p>לומדים איך לבחור ולבנות גרפים נכונים לדאטה רפואי – עם דגש על חיזוי test_normal</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="value">200</div><div class="label">רשומות בדאטהסט</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="value">7</div><div class="label">משתנים רפואיים</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="value">5</div><div class="label">סוגי גרפים שנלמד</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="goal-box">
        <h3>🎯 המטרה שלנו</h3>
        <p>אנחנו עובדים עם דאטהסט של בדיקות דם ורוצים <strong>לחזות</strong> האם תוצאת הבדיקה תהיה תקינה או לא
        (העמודה <code>test_normal</code>).</p>
        <p>לפני שבונים מודל, חשוב <strong>להבין את הנתונים</strong> דרך גרפים – כדי לגלות אילו משתנים
        עשויים להיות שימושיים לחיזוי.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👀 מבט ראשון על הדאטה")
    st.dataframe(df.drop(columns=["test_label"]).head(10), use_container_width=True)

    st.markdown("### 📋 עמודות הדאטהסט")
    col_info = {
        "heart_rate": ("רציף", "💓", "דופק לב – פעימות לדקה"),
        "body_temp": ("רציף", "🌡️", "טמפרטורת גוף – מעלות צלסיוס"),
        "protein_level": ("רציף", "🧪", "רמת חלבון בדם"),
        "white_blood_cells": ("רציף", "🔬", "ספירת תאי דם לבנים"),
        "hemoglobin": ("רציף", "🩸", "רמת המוגלובין"),
        "age": ("בדיד", "👤", "גיל המטופל (שלם)"),
        "gender": ("בדיד", "⚧️", "מגדר – 0 או 1"),
        "test_normal": ("בדיד", "✅", "תוצאת הבדיקה – 1=תקין, 0=לא תקין (TARGET)"),
    }
    rows = []
    for col, (kind, icon, desc) in col_info.items():
        rows.append({"עמודה": col, "סוג": kind, "אייקון": icon, "תיאור": desc})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 – בדיד vs רציף
# ═══════════════════════════════════════════════════════
elif page == "📖 נתון בדיד vs רציף":
    st.markdown("""
    <div class="section-header">
        <h2>📖 נתון בדיד לעומת נתון רציף</h2>
        <p>ההבחנה הבסיסית שתעזור לך לבחור את הגרף הנכון</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div class="concept-box">
            <h3>🔢 נתון בדיד (Discrete / Categorical)</h3>
            <p>ערכים שאפשר <strong>לספור</strong> – אין ביניהם ערכים אמצעיים.</p>
            <ul>
                <li><strong>קטגוריאלי:</strong> מגדר (זכר/נקבה), סוג טיפול</li>
                <li><strong>שלם:</strong> מספר ילדים, גיל (שנים שלמות)</li>
                <li><strong>בינארי:</strong> test_normal – 0 או 1 בלבד</li>
            </ul>
            <p>📊 <strong>גרפים מומלצים:</strong> Bar Chart, Pie Chart, Count Plot</p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="concept-box">
            <h3>📏 נתון רציף (Continuous)</h3>
            <p>ערכים שיכולים לקבל <strong>כל ערך</strong> בתחום נתון – יש ביניהם אינסוף ערכים אפשריים.</p>
            <ul>
                <li><strong>מדידות:</strong> טמפרטורת גוף – 36.5, 36.7, 37.1...</li>
                <li><strong>כמויות:</strong> דופק, רמת חלבון, המוגלובין</li>
                <li><strong>ספירות גדולות:</strong> תאי דם לבנים</li>
            </ul>
            <p>📈 <strong>גרפים מומלצים:</strong> Histogram, Box Plot, Violin Plot</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 📌 הדאטהסט שלנו – מה בדיד ומה רציף?")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f8fafc')

    # רציף – היסטוגרמה
    axes[0].hist(df["heart_rate"], bins=20, color="#2d6a9f", edgecolor="white", alpha=0.85)
    axes[0].set_title("דופק לב – נתון רציף\n(היסטוגרמה)", fontsize=13, pad=10)
    axes[0].set_xlabel("דופק")
    axes[0].set_ylabel("תדירות")
    axes[0].set_facecolor("#f0f7ff")

    # בדיד – bar
    counts = df["test_normal"].value_counts().rename(index={1: "תקין (1)", 0: "לא תקין (0)"})
    axes[1].bar(counts.index, counts.values, color=["#27ae60", "#e74c3c"], edgecolor="white", width=0.5)
    axes[1].set_title("test_normal – נתון בדיד\n(Bar Chart)", fontsize=13, pad=10)
    axes[1].set_ylabel("מספר מקרים")
    axes[1].set_facecolor("#f0f7ff")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="tip-box">
        <strong>💡 טיפ מהיר לזיהוי:</strong>
        שאל את עצמך – <em>"האם יכול להיות ערך של 1.5?"</em>
        <br>אם לא (כמו מגדר או test_normal) – זה בדיד. אם כן (כמו דופק=72.5) – זה רציף.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗂️ סיכום – איזה גרף לאיזה סוג?")
    summary_data = {
        "סוג נתון": ["בדיד / קטגוריאלי", "בדיד / קטגוריאלי", "רציף", "רציף", "רציף"],
        "גרף": ["Bar Chart", "Pie Chart", "Histogram", "Box Plot", "Violin Plot"],
        "שימוש עיקרי": [
            "להשוות כמויות בין קטגוריות",
            "להציג חלקים מהשלם (עד 5 קטגוריות)",
            "לראות את התפלגות הערכים",
            "להשוות התפלגויות + outliers",
            "Box Plot + צפיפות הנתונים",
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 – גרפים לנתון בדיד
# ═══════════════════════════════════════════════════════
elif page == "📊 גרפים לנתון בדיד":
    st.markdown("""
    <div class="section-header">
        <h2>📊 גרפים לנתון בדיד</h2>
        <p>Bar Chart ו-Pie Chart – מתי ואיך להשתמש בהם</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Bar Chart ──────────────────────────────────────
    st.markdown("## 1️⃣ Bar Chart – גרף עמודות")
    st.markdown("**מתי?** כשרוצים להשוות את **הכמות** בין קטגוריות שונות.")

    st.markdown('<div class="code-label">🐍 קוד Python – Bar Chart עם Matplotlib</div>', unsafe_allow_html=True)
    st.code("""
import matplotlib.pyplot as plt

counts = df["test_normal"].value_counts()
counts.index = ["תקין (1)", "לא תקין (0)"]   # שמות קריאים

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values, color=["#27ae60", "#e74c3c"])
ax.set_title("התפלגות תוצאות בדיקה")
ax.set_ylabel("מספר מקרים")
st.pyplot(fig)
""", language="python")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["test_normal"].value_counts().rename(index={1: "תקין", 0: "לא תקין"})
    ax.bar(counts.index, counts.values, color=["#27ae60", "#e74c3c"], edgecolor="white", width=0.5)
    ax.set_title("התפלגות תוצאות בדיקה", fontsize=13)
    ax.set_ylabel("מספר מקרים")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown("""
    <div class="goal-box">
        <strong>🎯 לחיזוי:</strong> אם יש חוסר איזון גדול בין הקלאסות (למשל 90% תקין, 10% לא תקין),
        המודל עלול להיות מוטה. Bar Chart מגלה זאת מיד!
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Bar Chart מקובץ ──────────────────────────────────
    st.markdown("## 2️⃣ Bar Chart מקובץ – לפי מגדר ותוצאה")
    st.markdown("**מתי?** כשרוצים להשוות שני משתנים בדידים **יחד**.")

    st.markdown('<div class="code-label">🐍 קוד Python – Bar Chart מקובץ</div>', unsafe_allow_html=True)
    st.code("""
grouped = df.groupby(["gender", "test_normal"]).size().unstack()
grouped.index = ["נקבה (0)", "זכר (1)"]
grouped.columns = ["לא תקין", "תקין"]

fig, ax = plt.subplots()
grouped.plot(kind="bar", ax=ax, color=["#e74c3c", "#27ae60"], edgecolor="white")
ax.set_title("תוצאת בדיקה לפי מגדר")
ax.set_xticklabels(grouped.index, rotation=0)
ax.set_ylabel("מספר מקרים")
ax.legend(title="תוצאה")
st.pyplot(fig)
""", language="python")

    grouped = df.groupby(["gender", "test_label"]).size().unstack(fill_value=0)
    grouped.index = ["נקבה (0)", "זכר (1)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.plot(kind="bar", ax=ax, color=["#e74c3c", "#27ae60"], edgecolor="white")
    ax.set_title("תוצאת בדיקה לפי מגדר", fontsize=13)
    ax.set_xticklabels(grouped.index, rotation=0)
    ax.set_ylabel("מספר מקרים")
    ax.legend(title="תוצאה")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Pie Chart ──────────────────────────────────────
    st.markdown("## 3️⃣ Pie Chart – תרשים עוגה")
    st.markdown("**מתי?** כשרוצים לראות את **הפרופורציה** מתוך השלם (מומלץ עד 4-5 קטגוריות).")

    st.markdown('<div class="code-label">🐍 קוד Python – Pie Chart</div>', unsafe_allow_html=True)
    st.code("""
counts = df["test_normal"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    counts.values,
    labels=["תקין", "לא תקין"],
    colors=["#27ae60", "#e74c3c"],
    autopct="%1.1f%%",       # מציג את האחוז
    startangle=90
)
ax.set_title("חלוקת תוצאות הבדיקה")
st.pyplot(fig)
""", language="python")

    counts = df["test_normal"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=["תקין", "לא תקין"],
           colors=["#27ae60", "#e74c3c"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("חלוקת תוצאות הבדיקה", fontsize=13)
    fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown("""
    <div class="tip-box">
        <strong>⚠️ מתי לא להשתמש ב-Pie Chart?</strong>
        כשיש יותר מ-5 קטגוריות, או כשהערכים קרובים מאוד זה לזה –
        קשה לעין להבחין בהבדלים. במקרה כזה, עדיף Bar Chart.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 4 – גרפים לנתון רציף
# ═══════════════════════════════════════════════════════
elif page == "📈 גרפים לנתון רציף":
    st.markdown("""
    <div class="section-header">
        <h2>📈 גרפים לנתון רציף</h2>
        <p>Histogram, Box Plot ו-Violin Plot – מתי ואיך להשתמש בהם</p>
    </div>
    """, unsafe_allow_html=True)

    continuous_cols = ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin"]
    col_names_heb = {
        "heart_rate": "דופק לב",
        "body_temp": "טמפרטורת גוף",
        "protein_level": "רמת חלבון",
        "white_blood_cells": "תאי דם לבנים",
        "hemoglobin": "המוגלובין"
    }
    selected_col = st.selectbox("בחר משתנה להצגה:", continuous_cols,
                                 format_func=lambda x: col_names_heb[x])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Histogram ──────────────────────────────────────
    st.markdown("## 1️⃣ Histogram – היסטוגרמה")
    st.markdown("**מה הוא מראה?** את **ההתפלגות** של הנתון – איפה מרוכזים הערכים, האם יש זנב ארוך, ועוד.")

    bins = st.slider("מספר bins:", 5, 50, 20)

    st.markdown('<div class="code-label">🐍 קוד Python – Histogram</div>', unsafe_allow_html=True)
    st.code(f"""
fig, ax = plt.subplots()
ax.hist(df["{selected_col}"], bins={bins}, color="#2d6a9f", edgecolor="white", alpha=0.85)
ax.set_title("התפלגות {col_names_heb[selected_col]}")
ax.set_xlabel("{selected_col}")
ax.set_ylabel("תדירות")
st.pyplot(fig)
""", language="python")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[selected_col], bins=bins, color="#2d6a9f", edgecolor="white", alpha=0.85)
    ax.set_title(f"התפלגות {col_names_heb[selected_col]}", fontsize=13)
    ax.set_xlabel(selected_col)
    ax.set_ylabel("תדירות")
    ax.set_facecolor("#f0f7ff")
    fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Box Plot ──────────────────────────────────────
    st.markdown("## 2️⃣ Box Plot – תרשים קופסה")
    st.markdown("""
    **מה הוא מראה?**
    - **קו אמצע** = חציון (מדיאן)
    - **הקופסה** = 50% האמצעיים של הנתונים (IQR)
    - **השפמים** = טווח הערכים הרגילים
    - **נקודות בודדות** = ערכים חריגים (Outliers)
    """)

    st.markdown('<div class="code-label">🐍 קוד Python – Box Plot לפי קבוצות</div>', unsafe_allow_html=True)
    st.code(f"""
fig, ax = plt.subplots()

groups = [
    df[df["test_normal"] == 1]["{selected_col}"],
    df[df["test_normal"] == 0]["{selected_col}"]
]
ax.boxplot(groups, labels=["תקין", "לא תקין"], patch_artist=True,
           boxprops=dict(facecolor="#aed6f1"),
           medianprops=dict(color="#e74c3c", linewidth=2))
ax.set_title("{col_names_heb[selected_col]} לפי תוצאת בדיקה")
ax.set_ylabel("{selected_col}")
st.pyplot(fig)
""", language="python")

    groups = [
        df[df["test_normal"] == 1][selected_col],
        df[df["test_normal"] == 0][selected_col]
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(groups, labels=["תקין (1)", "לא תקין (0)"], patch_artist=True,
                    boxprops=dict(facecolor="#aed6f1", linewidth=1.5),
                    medianprops=dict(color="#e74c3c", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=5, alpha=0.5))
    ax.set_title(f"{col_names_heb[selected_col]} לפי תוצאת בדיקה", fontsize=13)
    ax.set_ylabel(selected_col)
    ax.set_facecolor("#f0f7ff")
    fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown("""
    <div class="goal-box">
        <strong>🎯 לחיזוי:</strong> אם ה-Box Plot מראה הפרדה ברורה בין "תקין" ל"לא תקין",
        זהו סימן שהמשתנה יהיה <strong>מנבא טוב</strong> במודל!
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 5 – גרפים לחיזוי test_normal
# ═══════════════════════════════════════════════════════
elif page == "🎯 גרפים לחיזוי test_normal":
    st.markdown("""
    <div class="section-header">
        <h2>🎯 גרפים לחיזוי test_normal</h2>
        <p>כיצד ניתן להשתמש בגרפים כדי להבין אילו משתנים יעזרו לחיזוי</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="goal-box">
        <strong>🤔 מה אנחנו מחפשים?</strong>
        אנחנו רוצים למצוא משתנים שבהם <strong>יש הבדל ברור</strong> בין קבוצת "תקין" לקבוצת "לא תקין".
        ככל שההבדל גדול יותר – כך המשתנה שימושי יותר לחיזוי.
    </div>
    """, unsafe_allow_html=True)

    # ── השוואת התפלגויות ──────────────────────────────
    st.markdown("## 1️⃣ היסטוגרמות משולבות – הפרדה בין הקבוצות")
    st.markdown("מציגים את ההתפלגות של כל משתנה **בנפרד לכל קבוצה** – כדי לראות אם יש הפרדה.")

    st.markdown('<div class="code-label">🐍 קוד Python – היסטוגרמה משולבת</div>', unsafe_allow_html=True)
    st.code("""
fig, ax = plt.subplots()

df[df["test_normal"] == 1]["heart_rate"].plot(kind="hist", ax=ax,
    bins=20, alpha=0.6, color="#27ae60", label="תקין")
df[df["test_normal"] == 0]["heart_rate"].plot(kind="hist", ax=ax,
    bins=20, alpha=0.6, color="#e74c3c", label="לא תקין")

ax.set_title("דופק לב לפי תוצאת בדיקה")
ax.set_xlabel("heart_rate")
ax.legend()
st.pyplot(fig)
""", language="python")

    continuous_cols = ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor("#f8fafc")
    axes = axes.flatten()

    col_labels = {
        "heart_rate": "דופק לב",
        "body_temp": "טמפרטורת גוף",
        "protein_level": "רמת חלבון",
        "white_blood_cells": "תאי דם לבנים",
        "hemoglobin": "המוגלובין"
    }

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        df[df["test_normal"] == 1][col].plot(kind="hist", ax=ax,
            bins=20, alpha=0.6, color="#27ae60", label="תקין")
        df[df["test_normal"] == 0][col].plot(kind="hist", ax=ax,
            bins=20, alpha=0.6, color="#e74c3c", label="לא תקין")
        ax.set_title(col_labels[col], fontsize=12)
        ax.set_facecolor("#f0f7ff")
        ax.legend(fontsize=8)

    axes[5].set_visible(False)
    plt.suptitle("השוואת התפלגויות – תקין vs לא תקין", fontsize=14, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="tip-box">
        <strong>💡 מה לחפש?</strong> גרפים שבהם הירוק והאדום <strong>מופרדים</strong> (לא חופפים הרבה)
        מייצגים משתנים שימושיים לחיזוי. גרפים שבהם הם חופפים לחלוטין = משתנה פחות שימושי.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Box Plots ──────────────────────────────────────
    st.markdown("## 2️⃣ Box Plots לכל המשתנים – השוואה מהירה")
    st.markdown("רואים בבת אחת אילו משתנים מראים הבדל בין הקבוצות.")

    st.markdown('<div class="code-label">🐍 קוד Python – Box Plots מרובים</div>', unsafe_allow_html=True)
    st.code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for i, col in enumerate(["heart_rate", "body_temp", "hemoglobin"]):
    groups = [
        df[df["test_normal"] == 1][col],
        df[df["test_normal"] == 0][col]
    ]
    axes[i].boxplot(groups, labels=["תקין", "לא תקין"], patch_artist=True)
    axes[i].set_title(col)

st.pyplot(fig)
""", language="python")

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.patch.set_facecolor("#f8fafc")
    for i, col in enumerate(continuous_cols):
        groups = [
            df[df["test_normal"] == 1][col],
            df[df["test_normal"] == 0][col]
        ]
        bp = axes[i].boxplot(groups, labels=["תקין", "לא תקין"], patch_artist=True,
                             boxprops=dict(facecolor="#aed6f1"),
                             medianprops=dict(color="#e74c3c", linewidth=2),
                             flierprops=dict(marker='o', markersize=4, alpha=0.4))
        axes[i].set_title(col_labels[col], fontsize=11)
        axes[i].set_facecolor("#f0f7ff")
        axes[i].tick_params(axis='x', labelsize=8)

    plt.suptitle("Box Plots – כל המשתנים לפי תוצאת בדיקה", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Bar – גיל ──────────────────────────────────────
    st.markdown("## 3️⃣ גיל לפי תוצאת בדיקה – Bar Chart עם ממוצע")

    st.markdown('<div class="code-label">🐍 קוד Python – ממוצע לפי קבוצה</div>', unsafe_allow_html=True)
    st.code("""
avg_age = df.groupby("test_normal")["age"].mean()
avg_age.index = ["לא תקין (0)", "תקין (1)"]

fig, ax = plt.subplots()
ax.bar(avg_age.index, avg_age.values, color=["#e74c3c", "#27ae60"])
ax.set_title("גיל ממוצע לפי תוצאת בדיקה")
ax.set_ylabel("גיל ממוצע")
# מוסיפים את הערך מעל כל עמודה
for i, v in enumerate(avg_age.values):
    ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')
st.pyplot(fig)
""", language="python")

    avg_age = df.groupby("test_label")["age"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(avg_age.index, avg_age.values,
                  color=["#e74c3c" if x == "לא תקין" else "#27ae60" for x in avg_age.index],
                  edgecolor="white", width=0.5)
    ax.set_title("גיל ממוצע לפי תוצאת בדיקה", fontsize=13)
    ax.set_ylabel("גיל ממוצע")
    ax.set_facecolor("#f0f7ff")
    fig.patch.set_facecolor("#f8fafc")
    for i, (label, v) in enumerate(avg_age.items()):
        ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold', fontsize=11)
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════
# PAGE 6 – קורלציות
# ═══════════════════════════════════════════════════════
elif page == "🔗 קורלציות בין משתנים":
    st.markdown("""
    <div class="section-header">
        <h2>🔗 קורלציות בין משתנים</h2>
        <p>Heatmap ו-Scatter Plot – לגלות קשרים בין משתנים</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 1️⃣ Correlation Heatmap – מפת חום")
    st.markdown("""
    **מה הוא מראה?** הקורלציה (מתאם) בין כל זוג משתנים.
    - **1.0** = קורלציה חיובית מושלמת
    - **-1.0** = קורלציה שלילית מושלמת
    - **0** = אין קשר
    """)

    st.markdown('<div class="code-label">🐍 קוד Python – Heatmap עם Seaborn</div>', unsafe_allow_html=True)
    st.code("""
import seaborn as sns

corr = df.drop(columns=["test_label"]).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, center=0)
ax.set_title("מטריצת קורלציות")
st.pyplot(fig)
""", language="python")

    numeric_df = df.drop(columns=["test_label"])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, center=0,
                annot_kws={"size": 10})
    ax.set_title("מטריצת קורלציות", fontsize=13, pad=12)
    fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="goal-box">
        <strong>🎯 לחיזוי:</strong> שורה/עמודה של <code>test_normal</code> מראה אילו משתנים
        <strong>הכי קשורים לתוצאה</strong>. ערך גבוה (חיובי או שלילי) = מנבא פוטנציאלי חזק!
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Scatter Plot ──────────────────────────────────
    st.markdown("## 2️⃣ Scatter Plot – תרשים פיזור")
    st.markdown("**מה הוא מראה?** הקשר בין שני משתנים רציפים – ניתן לצבוע לפי הקבוצה.")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("ציר X:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"], index=0)
    with col2:
        y_col = st.selectbox("ציר Y:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"], index=4)

    st.markdown('<div class="code-label">🐍 קוד Python – Scatter Plot עם צביעה</div>', unsafe_allow_html=True)
    st.code(f"""
fig, ax = plt.subplots()

for label, color in [("תקין", "#27ae60"), ("לא תקין", "#e74c3c")]:
    subset = df[df["test_label"] == label]
    ax.scatter(subset["{x_col}"], subset["{y_col}"],
               label=label, color=color, alpha=0.6, s=40)

ax.set_xlabel("{x_col}")
ax.set_ylabel("{y_col}")
ax.set_title(f"{x_col} vs {y_col} לפי תוצאת בדיקה")
ax.legend()
st.pyplot(fig)
""", language="python")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in [("תקין", "#27ae60"), ("לא תקין", "#e74c3c")]:
        subset = df[df["test_label"] == label]
        ax.scatter(subset[x_col], subset[y_col],
                   label=label, color=color, alpha=0.55, s=40, edgecolors="white", linewidths=0.3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} לפי תוצאת בדיקה", fontsize=13)
    ax.legend()
    ax.set_facecolor("#f0f7ff")
    fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig)

    st.markdown("""
    <div class="tip-box">
        <strong>💡 מה לחפש?</strong> אם הנקודות הירוקות והאדומות <strong>מתפצלות לאזורים שונים</strong>
        בגרף – יש פוטנציאל להפריד ביניהן. זה בדיוק מה שמודלים כמו Decision Tree ו-SVM עושים!
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── סיכום ──────────────────────────────────────────
    st.markdown("## 📌 סיכום – טבלת הגרפים המלאה")
    summary = {
        "גרף": ["Bar Chart", "Pie Chart", "Histogram", "Box Plot", "Scatter Plot", "Heatmap"],
        "סוג נתון": ["בדיד", "בדיד", "רציף", "רציף", "רציף × רציף", "כל המשתנים"],
        "שימוש לחיזוי": [
            "לבדוק איזון בין קלאסות",
            "לראות פרופורציה של target",
            "לראות הפרדה בין קבוצות",
            "להשוות ממוצע + outliers בין קבוצות",
            "לגלות קשרים בין שני מנבאים",
            "למצוא מנבאים חזקים ל-test_normal"
        ]
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
