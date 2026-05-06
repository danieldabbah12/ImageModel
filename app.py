import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="מדריך ויזואליזציה | Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Assistant', sans-serif;
    direction: rtl;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
    direction: rtl;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 4px 0;
}

/* Main */
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem; }
.hero p  { font-size: 1.1rem; color: #94a3b8; margin: 0; }

/* Section title */
.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1.2rem;
    direction: rtl;
}

/* Cards */
.card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    direction: rtl;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 0.5rem;
}
.card p { color: #334155; font-size: 0.97rem; margin: 0; line-height: 1.7; }

/* When-to-use box */
.when-box {
    background: #eff6ff;
    border-right: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    direction: rtl;
}
.when-box strong { color: #1d4ed8; font-size: 1rem; }
.when-box ul { margin: 0.4rem 0 0 0; padding-right: 1.2rem; color: #334155; }
.when-box li { margin-bottom: 0.25rem; font-size: 0.95rem; }

/* Params box */
.param-box {
    background: #f0fdf4;
    border-right: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    direction: rtl;
}
.param-box strong { color: #15803d; font-size: 1rem; }
.param-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.9rem; }
.param-table th { background: #dcfce7; color: #166534; padding: 6px 10px; text-align: right; }
.param-table td { padding: 6px 10px; border-bottom: 1px solid #bbf7d0; color: #334155; vertical-align: top; }
.param-table tr:last-child td { border-bottom: none; }

/* Code block override — LTR always */
.stCodeBlock, code, pre {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: embed;
}

/* Divider */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }

/* Badge */
.badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
@st.cache_data
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
    df["heart_rate"]        = df["heart_rate"].round(0)
    df["body_temp"]         = df["body_temp"].round(1)
    df["protein_level"]     = df["protein_level"].round(2)
    df["white_blood_cells"] = df["white_blood_cells"].round(0)
    df["hemoglobin"]        = df["hemoglobin"].round(1)
    return df

df = generate_blood_test_dataset()
df["test_label"] = df["test_normal"].map({1: "תקין", 0: "לא תקין"})


# ─────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────
PAGES = {
    "🏠 דף הבית":          "home",
    "📊 Histogram":         "histogram",
    "📊 Bar Chart":         "bar_chart",
    "📦 Box Plot":          "box_plot",
    "🔵 Scatter Plot":      "scatter",
    "🔥 Heatmap (Corr.)":   "heatmap",
    "🥧 Pie / Donut Chart": "pie",
    "📋 Data Summary":      "summary",
}

with st.sidebar:
    st.markdown("## 📊 מדריך ויזואליזציה")
    st.markdown("---")
    page_label = st.radio("בחר נושא:", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#94a3b8; line-height:1.7;'>
    הדאטהסט: בדיקות דם<br>
    100 תצפיות · 7 עמודות<br>
    משתנה מטרה: <code>test_normal</code>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def card(title, body):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div><p>{body}</p></div>', unsafe_allow_html=True)

def when_box(bullets: list):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"""
    <div class="when-box">
      <strong>✅ מתי נשתמש בגרף זה?</strong>
      <ul>{items}</ul>
    </div>""", unsafe_allow_html=True)

def param_table(rows: list):
    """rows = [(param, type, description)]"""
    header = "<tr><th>פרמטר</th><th>סוג</th><th>תיאור</th></tr>"
    body   = "".join(f"<tr><td><code>{r[0]}</code></td><td><code>{r[1]}</code></td><td>{r[2]}</td></tr>" for r in rows)
    st.markdown(f"""
    <div class="param-box">
      <strong>⚙️ פרמטרים חשובים</strong>
      <table class="param-table">{header}{body}</table>
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>📊 מדריך ויזואליזציה ב-Streamlit</h1>
      <p>cheatsheet מלא לבניית גרפים — צעד אחר צעד, עם הסברים בעברית</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🩺 הדאטהסט שבו נשתמש לאורך המדריך")
    card("בדיקות דם — Blood Test Dataset",
         "הדאטהסט מכיל 100 תצפיות של בדיקות דם. כל שורה מייצגת מטופל אחד. "
         "המשתנים הם: <b>heart_rate</b> (דופק), <b>body_temp</b> (חום גוף), "
         "<b>protein_level</b> (רמת חלבון), <b>white_blood_cells</b> (כדוריות דם לבנות), "
         "<b>hemoglobin</b> (המוגלובין), <b>age</b> (גיל), <b>gender</b> (מגדר 0/1), "
         "ו-<b>test_normal</b> (1 = תקין, 0 = לא תקין).")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 👀 הצצה לדאטה")
        st.dataframe(df.drop(columns=["test_label"]).head(8), use_container_width=True)
    with col2:
        st.markdown("#### 📐 מבנה הדאטה")
        info_df = pd.DataFrame({
            "עמודה": df.drop(columns=["test_label"]).columns,
            "סוג": df.drop(columns=["test_label"]).dtypes.astype(str).values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    divider()
    st.markdown("### 🗺️ מה תמצאו במדריך?")

    cols = st.columns(3)
    guides = [
        ("📊", "Histogram", "התפלגות משתנה רציף"),
        ("📊", "Bar Chart", "השוואת ערכים קטגוריאליים"),
        ("📦", "Box Plot", "פיזור וחריגות"),
        ("🔵", "Scatter Plot", "קשר בין שני משתנים"),
        ("🔥", "Heatmap", "מטריצת מתאמים"),
        ("🥧", "Pie Chart", "חלוקת קטגוריות"),
    ]
    for i, (icon, name, desc) in enumerate(guides):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="text-align:center">
              <div style="font-size:2rem">{icon}</div>
              <div class="card-title" style="text-align:center">{name}</div>
              <p style="text-align:center">{desc}</p>
            </div>""", unsafe_allow_html=True)

    divider()
    st.info("💡 **טיפ:** השתמש בתפריט הצד כדי לנווט בין הגרפים השונים. לכל גרף תמצאי הסבר, דוגמת קוד, ופרמטרים.")


# ═══════════════════════════════════════════════════════════
# HISTOGRAM
# ═══════════════════════════════════════════════════════════
elif page == "histogram":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">📊 Histogram — היסטוגרמה</h1><p>הצגת התפלגות של משתנה רציף</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים לראות איך ערכים של משתנה רציף מתפזרים (לדוג׳: אילו ערכי דופק הם הנפוצים ביותר?)",
        "רוצים לבדוק האם ההתפלגות סימטרית, עם צניחה לצד אחד, או דו-שיאית",
        "רוצים להבין את טווח הערכים: מינימום, מקסימום, ריכוז",
    ])

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# df הוא ה-DataFrame שלך
st.subheader("התפלגות דופק הלב")
st.bar_chart(df["heart_rate"].value_counts().sort_index())
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>df["heart_rate"]</b> — בוחרים עמודה אחת מה-DataFrame.<br>
    <b>.value_counts()</b> — סופרת כמה פעמים מופיע כל ערך.<br>
    <b>.sort_index()</b> — ממיינת לפי ערך הציר ה-X (מקטן לגדול) כדי שהגרף יהיה קריא.<br>
    <b>st.bar_chart(...)</b> — מציגה את התוצאה כגרף עמודות (שמתנהג כהיסטוגרמה).
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        ("df[\"col\"]",       "Series",  "עמודה מתוך ה-DataFrame"),
        (".value_counts()",   "method",  "מחזירה Series עם ספירת ערכים"),
        (".sort_index()",     "method",  "ממיין לפי האינדקס (ציר X)"),
        ("st.bar_chart()",    "function","מציגה גרף עמודות / היסטוגרמה"),
    ])

    section("📈 הגרף בפועל")

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_col = st.selectbox("בחר עמודה:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"])
        show_by_label = st.checkbox("הצג לפי תוצאת בדיקה (תקין/לא תקין)")

    with col2:
        if not show_by_label:
            chart_data = df[chosen_col].value_counts().sort_index().rename("Count")
            st.bar_chart(chart_data)
        else:
            g0 = df[df["test_label"]=="תקין"][chosen_col].value_counts().sort_index().rename("תקין")
            g1 = df[df["test_label"]=="לא תקין"][chosen_col].value_counts().sort_index().rename("לא תקין")
            combined = pd.concat([g0, g1], axis=1).fillna(0)
            st.bar_chart(combined)

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    עבור <b>heart_rate</b>: רוב הערכים מרוכזים סביב 70–80 פעימות (קבוצה תקינה),
    ועמודה שנייה ניתן לראות סביב 85–100 (קבוצה לא תקינה). זה מלמד שיש שתי אוכלוסיות שונות בדאטה.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# BAR CHART
# ═══════════════════════════════════════════════════════════
elif page == "bar_chart":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">📊 Bar Chart — גרף עמודות</h1><p>השוואת ממוצעים בין קטגוריות</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים להשוות את הממוצע (או סכום) של משתנה רציף בין קבוצות (לדוג׳: ממוצע דופק — תקין לעומת לא תקין)",
        "רוצים לראות כמה נתונים יש בכל קטגוריה",
        "כל שורה בדאטה שייכת לקטגוריה ורוצים לצמצם אותה לערך אחד לקטגוריה",
    ])

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# השוואת ממוצע דופק בין תקין / לא תקין
avg_by_group = df.groupby("test_normal")["heart_rate"].mean()

st.subheader("ממוצע דופק לב לפי תוצאת בדיקה")
st.bar_chart(avg_by_group)
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>df.groupby("test_normal")</b> — מחלקת את ה-DataFrame לפי ערכי העמודה "test_normal".<br>
    <b>["heart_rate"]</b> — בוחרת את העמודה שרוצים לחשב עליה.<br>
    <b>.mean()</b> — מחשבת ממוצע לכל קבוצה. אפשר גם <code>.sum()</code>, <code>.count()</code>, <code>.median()</code>.<br>
    <b>st.bar_chart(...)</b> — מציגה את התוצאה כגרף עמודות.
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        ("df.groupby(col)",  "method",   "מחלק את הדאטה לקבוצות לפי עמודה"),
        ("[col2]",           "indexing", "בוחר עמודה לביצוע חישוב עליה"),
        (".mean()",          "method",   "ממוצע — אפשר גם .sum() / .count() / .median()"),
        ("st.bar_chart()",   "function", "גרף עמודות ב-Streamlit"),
        ("use_container_width", "bool", "האם הגרף ימלא את רוחב הדף (True/False)"),
    ])

    section("📈 הגרף בפועל")

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_col = st.selectbox("בחר משתנה רציף:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin"])
        agg_func    = st.selectbox("פעולת צבירה:", ["mean", "median", "max", "min"])
        group_col   = st.selectbox("קבץ לפי:", ["test_normal", "gender"])

    with col2:
        agg = df.groupby(group_col)[chosen_col].agg(agg_func)
        agg.index = agg.index.astype(str)
        st.bar_chart(agg, use_container_width=True)

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    כאשר מקבצים לפי <b>test_normal</b> ובוחרים <b>mean</b> של <b>heart_rate</b>, רואים בבירור
    שלמטופלים לא תקינים (0) יש ממוצע דופק גבוה משמעותית לעומת קבוצה תקינה (1).
    זהו גרף נהדר לזיהוי הבדלים בין קבוצות.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# BOX PLOT
# ═══════════════════════════════════════════════════════════
elif page == "box_plot":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">📦 Box Plot — תיבת שפם</h1><p>פיזור נתונים, חציון וחריגות</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים לראות בו זמנית: מינימום, Q1, חציון (median), Q3, מקסימום — והכל בגרף אחד",
        "רוצים לזהות outliers (ערכים חריגים) — הנקודות שמחוץ לשפם",
        "רוצים להשוות פיזור של משתנה רציף בין מספר קבוצות",
    ])

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# יצירת pivot: עמודה לכל קבוצה
pivot = df.groupby("test_normal")["heart_rate"].apply(list)
box_df = pd.DataFrame({
    "Normal (1)":   pivot[1],
    "Abnormal (0)": pivot[0],
})

st.subheader("פיזור דופק לפי תוצאת בדיקה")

# Streamlit לא תומך ב-box plot ישיר — נשתמש ב-line_chart על פני percentiles
desc = df.groupby("test_normal")["heart_rate"].describe()[["25%","50%","75%"]]
st.bar_chart(desc.T)
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>Streamlit</b> לא כולל <code>box_plot</code> מובנה, אבל אפשר לחקות אותו!<br>
    <b>.describe()</b> — מחזיר סטטיסטיקות: mean, std, min, 25%, 50% (חציון), 75%, max.<br>
    <b>[["25%","50%","75%"]]</b> — בוחרים רק את הרביעונים.<br>
    <b>.T</b> — Transpose: מחליף שורות ועמודות, כדי שהגרף יראה נכון.<br>
    <b>st.bar_chart(desc.T)</b> — מציגה גרף שמדגים את הפיזור.
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        (".describe()",        "method",   "מחזיר טבלת סטטיסטיקות: min/max/mean/quartiles"),
        ("[['25%','50%','75%']]", "indexing","בחירת עמודות ספציפיות מהתוצאה"),
        (".T",                 "attribute","Transpose — הופך שורות לעמודות ולהיפך"),
        ("st.bar_chart()",     "function", "גרף עמודות מקובצות לפי שם עמודה"),
    ])

    section("📈 הגרף בפועל")

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_col = st.selectbox("בחר משתנה:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin"])
        show_all   = st.checkbox("הצג גם ממוצע ומינימום/מקסימום")
        stats_cols = ["25%","50%","75%"] if not show_all else ["min","25%","50%","75%","max","mean"]

    with col2:
        desc = df.groupby("test_normal")[chosen_col].describe()[stats_cols]
        desc.index = desc.index.map({0: "Abnormal (0)", 1: "Normal (1)"})
        st.bar_chart(desc.T, use_container_width=True)
        st.caption("כל קבוצת עמודות = קבוצה אחת (תקין/לא תקין). גובה העמודה = ערך האחוזון.")

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    ה-<b>50%</b> הוא החציון — ערך ה"אמצעי" שמחלק את הדאטה לשניים.
    ה-<b>25% עד 75%</b> הם ה-IQR (Interquartile Range) — הרוב של הנתונים נמצאים שם.
    ערכים מחוץ לטווח זה הם חשודים כ-Outliers.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SCATTER PLOT
# ═══════════════════════════════════════════════════════════
elif page == "scatter":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">🔵 Scatter Plot — גרף פיזור</h1><p>קשר בין שני משתנים רציפים</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים לראות האם שני משתנים רציפים קשורים אחד לשני (קורלציה)",
        "רוצים לזהות דפוסים, אשכולות, או ערכים חריגים",
        "שאלה כמו: 'האם מי שיש לו דופק גבוה גם סובל מחום גבוה?'",
    ])

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# גרף פיזור: ציר X = heart_rate, ציר Y = body_temp
scatter_df = df[["heart_rate", "body_temp"]]

st.subheader("קשר בין דופק לחום גוף")
st.scatter_chart(scatter_df, x="heart_rate", y="body_temp")
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>st.scatter_chart()</b> — פונקציה ייעודית ב-Streamlit לגרף פיזור.<br>
    <b>x="heart_rate"</b> — שם העמודה שתוצג בציר האופקי (X).<br>
    <b>y="body_temp"</b> — שם העמודה שתוצג בציר האנכי (Y).<br>
    <b>color="test_normal"</b> — ניתן לצבוע את הנקודות לפי קטגוריה (ראו דוגמה בגרף למטה).
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        ("st.scatter_chart()", "function", "גרף פיזור מובנה ב-Streamlit"),
        ("x",                  "str",      "שם עמודת ציר X (חובה)"),
        ("y",                  "str/list", "שם עמודת ציר Y — אפשר גם רשימה"),
        ("color",              "str",      "עמודה לפי שתצביע את הנקודות"),
        ("size",               "str",      "עמודה שתקבע את גודל הנקודה"),
        ("use_container_width","bool",     "האם הגרף ימלא את רוחב הדף"),
    ])

    section("📈 הגרף בפועל")

    col1, col2 = st.columns([1, 2])
    with col1:
        x_col   = st.selectbox("ציר X:", ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"])
        y_col   = st.selectbox("ציר Y:", ["body_temp", "heart_rate", "protein_level", "white_blood_cells", "hemoglobin", "age"])
        color_col = st.selectbox("צבע לפי:", ["test_normal", "gender", "— ללא —"])

    with col2:
        scatter_df = df[[x_col, y_col]].copy()
        if color_col != "— ללא —":
            scatter_df[color_col] = df[color_col].astype(str)
            st.scatter_chart(scatter_df, x=x_col, y=y_col, color=color_col, use_container_width=True)
        else:
            st.scatter_chart(scatter_df, x=x_col, y=y_col, use_container_width=True)

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    נסו לבחור <b>X = heart_rate, Y = body_temp</b> ולצבוע לפי <b>test_normal</b>.
    תראו שתי ענני נקודות — אחד לתקינים ואחד ללא תקינים — שמפוזרים באזורים שונים.
    זה מראה שיש קשר בין שני המשתנים לתוצאת הבדיקה.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HEATMAP
# ═══════════════════════════════════════════════════════════
elif page == "heatmap":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">🔥 Heatmap — מפת חום</h1><p>מטריצת מתאמים בין כל המשתנים</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים לראות בבת אחת את כל הקשרים בין כל הזוגות של משתנים",
        "רוצים לבדוק אילו משתנים קשורים חזק למשתנה המטרה (test_normal)",
        "רוצים לזהות multicollinearity — שני משתנים שמכילים אותו מידע",
    ])

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# חישוב מטריצת מתאמים
numeric_cols = ["heart_rate", "body_temp", "protein_level",
                "white_blood_cells", "hemoglobin", "age", "test_normal"]
corr_matrix = df[numeric_cols].corr()

st.subheader("מטריצת מתאמים")
st.dataframe(
    corr_matrix.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
    use_container_width=True
)
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>.corr()</b> — מחשב את מקדם הקורלציה של פירסון בין כל זוג עמודות. ערכים בין -1 ל-1.<br>
    <b>st.dataframe()</b> — מציגה DataFrame אינטראקטיבי.<br>
    <b>.style.background_gradient()</b> — מוסיף צבע לפי ערך: ירוק = קורלציה חיובית, אדום = שלילית.<br>
    <b>cmap="RdYlGn"</b> — פלטת הצבעים: Red-Yellow-Green.
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        (".corr()",                "method",   "מחשב מטריצת קורלציה (Pearson)"),
        (".style.background_gradient()", "method", "מוסיף צביעה תלוית ערך לטבלה"),
        ("cmap",                   "str",      "שם פלטת הצבעים: 'RdYlGn', 'coolwarm', 'Blues'..."),
        ("vmin / vmax",            "float",    "ערך מינימום/מקסימום לסקאלת הצבע"),
        ("st.dataframe()",         "function", "מציגה DataFrame — ניתן לגלול ולמיין"),
    ])

    section("📈 הגרף בפועל")

    numeric_cols = ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age", "test_normal"]
    corr = df[numeric_cols].corr()

    col1, col2 = st.columns([1, 2])
    with col1:
        cmap_choice = st.selectbox("פלטת צבעים:", ["RdYlGn", "coolwarm", "Blues", "Oranges"])
        st.markdown("#### מה המספרים אומרים?")
        st.markdown("""
        - **1.0** = קורלציה חיובית מושלמת
        - **0.0** = אין קשר
        - **-1.0** = קורלציה שלילית מושלמת
        """)
    with col2:
        st.dataframe(
            corr.style.background_gradient(cmap=cmap_choice, vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True
        )

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    שימו לב לעמודה/שורה של <b>test_normal</b> — שם תראו אילו משתנים קשורים ביותר לתוצאת הבדיקה.
    <b>heart_rate</b> ו-<b>white_blood_cells</b> אמורים להציג קורלציה שלילית (ביחס ל-test_normal=1).
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PIE CHART
# ═══════════════════════════════════════════════════════════
elif page == "pie":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">🥧 Pie Chart — גרף עוגה</h1><p>חלוקת קטגוריות לפי אחוזים</p></div>', unsafe_allow_html=True)

    when_box([
        "רוצים לראות איזה חלק מהנתונים שייך לכל קטגוריה (כמו: כמה % תקינים מכלל הדאטה?)",
        "המשתנה הוא קטגוריאלי (בדידי) עם מספר קטן של קטגוריות (2–6)",
        "חשוב להדגיש חלוקה יחסית ולא ערכים מוחלטים",
    ])

    st.markdown("""
    <div style="background:#fefce8; border-right:4px solid #eab308; border-radius:0 8px 8px 0; 
                padding:1rem 1.2rem; margin-bottom:1.2rem; direction:rtl;">
    <strong style="color:#92400e;">⚠️ מגבלה: Streamlit לא כולל pie chart מובנה</strong><br>
    <span style="color:#78350f; font-size:0.95rem;">
    ב-Streamlit הבסיסי אין <code>st.pie_chart()</code>. הדרך הקלה ביותר היא להציג את נתוני העוגה
    כ-<b>גרף עמודות אופקי</b> שנראה נקי וברור. נציג גם את ה-DataFrame עם האחוזים.
    </span>
    </div>
    """, unsafe_allow_html=True)

    section("🔧 דוגמת קוד")
    st.code("""
import streamlit as st
import pandas as pd

# ספירת קטגוריות + חישוב אחוזים
counts = df["test_normal"].value_counts()
pct    = (counts / counts.sum() * 100).round(1)

# DataFrame מסודר
summary = pd.DataFrame({
    "קטגוריה": ["תקין (1)", "לא תקין (0)"],
    "כמות":    counts.values,
    "אחוז (%)": pct.values
})

st.subheader("חלוקת תוצאות בדיקה")
st.dataframe(summary, hide_index=True)

# גרף עמודות אופקי
st.bar_chart(counts)
""", language="python")

    st.markdown("""
    <div class="card">
    <div class="card-title">💡 הסבר על הקוד</div>
    <p>
    <b>.value_counts()</b> — סופרת כמה פעמים מופיעה כל קטגוריה.<br>
    <b>/ counts.sum() * 100</b> — מחשבת אחוז מסך כל הנתונים.<br>
    <b>.round(1)</b> — מעגלת לספרה אחת אחרי הנקודה.<br>
    <b>st.dataframe()</b> — מציגה טבלה נקייה.<br>
    <b>hide_index=True</b> — מסתיר את עמודת האינדקס (0,1,2...) שאינה דרושה.
    </p>
    </div>
    """, unsafe_allow_html=True)

    param_table([
        (".value_counts()",    "method",  "ספירת כל קטגוריה"),
        (".sum()",             "method",  "סכום כל הערכים — לחישוב אחוזים"),
        ("st.dataframe()",     "function","הצגת טבלה אינטראקטיבית"),
        ("hide_index",         "bool",    "True = הסתר עמודת אינדקס"),
        ("st.bar_chart()",     "function","גרף עמודות — כחלופה לגרף עוגה"),
    ])

    section("📈 הגרף בפועל")

    col1, col2 = st.columns([1, 1])
    with col1:
        cat_col = st.selectbox("בחר עמודה קטגוריאלית:", ["test_normal", "gender"])
        counts = df[cat_col].value_counts()
        pct    = (counts / counts.sum() * 100).round(1)
        summary_df = pd.DataFrame({
            "קטגוריה": counts.index.astype(str),
            "כמות":    counts.values,
            "אחוז (%)":pct.values
        })
        st.markdown("#### טבלת סיכום")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### גרף עמודות")
        counts.index = counts.index.astype(str)
        st.bar_chart(counts, use_container_width=True)

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 מה ניתן ללמוד מהגרף?</div>
    <p>
    ניתן לראות שהדאטהסט <b>מאוזן</b> — כ-50% תקינים וכ-50% לא תקינים (עקב האופן שבו נוצר).
    בדאטה אמיתי, לעתים קרובות יש חוסר איזון (class imbalance) שיכול לבעל בעיות בלמידת מכונה.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA SUMMARY
# ═══════════════════════════════════════════════════════════
elif page == "summary":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem; text-align:right"><h1 style="font-size:1.8rem">📋 Data Summary — סיכום הדאטה</h1><p>כלים בסיסיים לחקר ראשוני של הדאטהסט</p></div>', unsafe_allow_html=True)

    when_box([
        "מתחילים לעבוד עם דאטהסט חדש וצריך להבין את מבנהו",
        "רוצים לבדוק ערכים חסרים, סוגי עמודות, ומספר תצפיות",
        "רוצים סטטיסטיקות בסיסיות לפני שבונים גרפים",
    ])

    section("🔧 דוגמאות קוד")

    tab1, tab2, tab3 = st.tabs(["📄 .head() / .tail()", "📊 .describe()", "🔍 .info() style"])

    with tab1:
        st.code("""
# הצגת 5 השורות הראשונות
st.write("### ראשית הדאטה")
st.dataframe(df.head())

# הצגת 5 השורות האחרונות
st.write("### סוף הדאטה")
st.dataframe(df.tail())

# הצגת מספר שורות לפי בחירת המשתמש
n = st.slider("כמה שורות להציג?", 5, 50, 10)
st.dataframe(df.head(n))
""", language="python")
        st.markdown("""
        <div class="card"><div class="card-title">💡 הסבר</div>
        <p><b>df.head(n)</b> — מחזיר את n השורות הראשונות (ברירת מחדל: 5).<br>
        <b>df.tail(n)</b> — מחזיר את n השורות האחרונות.<br>
        <b>st.slider()</b> — מאפשר למשתמש לבחור מספר ידנית.</p></div>
        """, unsafe_allow_html=True)

        n = st.slider("כמה שורות להציג?", 5, 30, 8)
        st.dataframe(df.drop(columns=["test_label"]).head(n), use_container_width=True)

    with tab2:
        st.code("""
# סטטיסטיקות תיאוריות לכל עמודה מספרית
st.write("### סטטיסטיקות")
st.dataframe(df.describe().T)

# .T = Transpose — הופך שורות לעמודות לקריאות נוחה
""", language="python")
        st.markdown("""
        <div class="card"><div class="card-title">💡 הסבר</div>
        <p><b>df.describe()</b> — מחשב: count, mean, std, min, 25%, 50%, 75%, max.<br>
        <b>.T</b> — Transpose: כדי שכל עמודת דאטה תהיה שורה — נוח יותר לקריאה.</p></div>
        """, unsafe_allow_html=True)

        st.dataframe(df.drop(columns=["test_label"]).describe().T.round(2), use_container_width=True)

    with tab3:
        st.code("""
# סוגי עמודות + כמות ערכים חסרים
info_df = pd.DataFrame({
    "dtype":    df.dtypes.astype(str),
    "non_null": df.notnull().sum(),
    "null":     df.isnull().sum(),
    "unique":   df.nunique(),
})
st.write("### מידע על עמודות")
st.dataframe(info_df)
""", language="python")
        st.markdown("""
        <div class="card"><div class="card-title">💡 הסבר</div>
        <p><b>df.dtypes</b> — סוג כל עמודה (int64, float64, object...).<br>
        <b>df.notnull().sum()</b> — כמה ערכים לא חסרים.<br>
        <b>df.isnull().sum()</b> — כמה ערכים חסרים (NaN).<br>
        <b>df.nunique()</b> — כמה ערכים ייחודיים בכל עמודה.</p></div>
        """, unsafe_allow_html=True)

        info_df = pd.DataFrame({
            "dtype":    df.drop(columns=["test_label"]).dtypes.astype(str),
            "non_null": df.drop(columns=["test_label"]).notnull().sum(),
            "null":     df.drop(columns=["test_label"]).isnull().sum(),
            "unique":   df.drop(columns=["test_label"]).nunique(),
        })
        st.dataframe(info_df, use_container_width=True)

    divider()
    st.markdown("""
    <div class="card">
    <div class="card-title">📌 טיפ: ה-EDA הבסיסי</div>
    <p>
    בכל פרויקט חדש, תמיד התחילו עם 3 צעדים:<br>
    1. <code>df.head()</code> — ראו איך נראה הדאטה<br>
    2. <code>df.describe()</code> — הבינו את הטווחים<br>
    3. <code>df.isnull().sum()</code> — בדקו ערכים חסרים<br>
    רק אחר כך עברו לגרפים!
    </p>
    </div>
    """, unsafe_allow_html=True)
