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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Assistant', sans-serif;
    direction: rtl;
}
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; direction: rtl; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1000px; }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 2rem;
    color: white;
    direction: rtl;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem; }
.hero p  { font-size: 1rem; color: #94a3b8; margin: 0; }

.section-title {
    font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.3rem; margin: 1.8rem 0 1rem;
    direction: rtl;
}

.when-box {
    background: #eff6ff;
    border-right: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem; direction: rtl;
}
.when-box strong { color: #1d4ed8; font-size: 1rem; }
.when-box ul { margin: 0.4rem 0 0; padding-right: 1.2rem; color: #334155; }
.when-box li { margin-bottom: 0.2rem; font-size: 0.95rem; }

.explain-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem; direction: rtl;
}
.explain-box b { color: #1e40af; }

.param-box {
    background: #f0fdf4; border-right: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem; direction: rtl;
}
.param-box strong { color: #15803d; }
.ptable { width:100%; border-collapse:collapse; margin-top:0.5rem; font-size:0.9rem; }
.ptable th { background:#dcfce7; color:#166534; padding:6px 10px; text-align:right; }
.ptable td { padding:6px 10px; border-bottom:1px solid #bbf7d0; color:#334155; vertical-align:top; }
.ptable tr:last-child td { border-bottom:none; }

.stCodeBlock, code, pre { direction: ltr !important; text-align: left !important; }
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
        "heart_rate":        np.random.normal(72, 8, half),
        "body_temp":         np.random.normal(36.6, 0.3, half),
        "protein_level":     np.random.normal(7.0, 0.5, half),
        "white_blood_cells": np.random.normal(7000, 1200, half),
        "hemoglobin":        np.random.normal(14, 1.5, half),
        "age":               np.random.randint(18, 65, half),
        "gender":            np.random.randint(0, 2, half),
        "test_normal":       1
    })
    abnormal = pd.DataFrame({
        "heart_rate":        np.random.normal(90, 15, half),
        "body_temp":         np.random.normal(37.8, 0.8, half),
        "protein_level":     np.random.normal(6.0, 1.0, half),
        "white_blood_cells": np.random.normal(11000, 3000, half),
        "hemoglobin":        np.random.normal(11.5, 2.0, half),
        "age":               np.random.randint(18, 80, half),
        "gender":            np.random.randint(0, 2, half),
        "test_normal":       0
    })
    df = pd.concat([normal, abnormal]).sample(frac=1).reset_index(drop=True)
    noise_idx = np.random.choice(n, int(n * noise_ratio), replace=False)
    df.loc[noise_idx, "test_normal"] = 1 - df.loc[noise_idx, "test_normal"]
    df["heart_rate"]        = df["heart_rate"].round(0)
    df["body_temp"]         = df["body_temp"].round(1)
    df["protein_level"]     = df["protein_level"].round(2)
    df["white_blood_cells"] = df["white_blood_cells"].round(0)
    df["hemoglobin"]        = df["hemoglobin"].round(1)
    return df

df = generate_blood_test_dataset()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
PAGES = {
    "🏠 דף הבית":      "home",
    "📊 Histogram":    "histogram",
    "📊 Bar Chart":    "bar_chart",
    "🔵 Scatter Plot": "scatter",
}

with st.sidebar:
    st.markdown("## 📊 מדריך ויזואליזציה")
    st.markdown("---")
    page_label = st.radio("בחר נושא:", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.82rem;color:#94a3b8;line-height:1.8'>"
        "דאטהסט: בדיקות דם<br>100 תצפיות · 8 עמודות<br>"
        "מטרה: <code>test_normal</code></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(t):
    st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)

def when_box(bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f'<div class="when-box"><strong>✅ מתי נשתמש בגרף זה?</strong><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )

def explain(body):
    st.markdown(f'<div class="explain-box">{body}</div>', unsafe_allow_html=True)

def param_table(rows):
    header = "<tr><th>פרמטר</th><th>סוג</th><th>תיאור</th></tr>"
    body   = "".join(
        f"<tr><td><code>{r[0]}</code></td><td><code>{r[1]}</code></td><td>{r[2]}</td></tr>"
        for r in rows
    )
    st.markdown(
        f'<div class="param-box"><strong>⚙️ פרמטרים</strong>'
        f'<table class="ptable">{header}{body}</table></div>',
        unsafe_allow_html=True,
    )

def divider():
    st.markdown("---")


# ═══════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>📊 מדריך ויזואליזציה ב-Streamlit</h1>
      <p>הסברים פשוטים לבניית גרפים — עם דוגמאות קוד וגרפים חיים</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🩺 הדאטהסט שבו נשתמש")
    explain("""
    הדאטהסט מכיל <b>100 רשומות של בדיקות דם</b>. כל שורה = מטופל אחד.<br><br>
    <b>משתנים רציפים (מספרים):</b><br>
    &nbsp;&nbsp;• <b>heart_rate</b> — דופק לב (פעימות לדקה)<br>
    &nbsp;&nbsp;• <b>body_temp</b> — חום גוף (מעלות צלזיוס)<br>
    &nbsp;&nbsp;• <b>protein_level</b> — רמת חלבון בדם<br>
    &nbsp;&nbsp;• <b>white_blood_cells</b> — כדוריות דם לבנות<br>
    &nbsp;&nbsp;• <b>hemoglobin</b> — המוגלובין<br>
    &nbsp;&nbsp;• <b>age</b> — גיל<br><br>
    <b>משתנים בדידים (קטגוריות):</b><br>
    &nbsp;&nbsp;• <b>gender</b> — מגדר (0 / 1)<br>
    &nbsp;&nbsp;• <b>test_normal</b> — תוצאת הבדיקה: <b>1 = תקין</b>, <b>0 = לא תקין</b>
    """)

    st.dataframe(df.head(8), use_container_width=True)

    st.markdown("### 🗺️ מה תמצאו במדריך?")
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📊", "Histogram", "התפלגות של משתנה רציף"),
        ("📊", "Bar Chart", "כמות תצפיות לפי קטגוריה"),
        ("🔵", "Scatter Plot", "קשר בין שני משתנים רציפים"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="explain-box" style="text-align:center">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<b>{title}</b><br>'
                f'<span style="color:#64748b;font-size:0.9rem">{desc}</span></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
# HISTOGRAM
# ═══════════════════════════════════════════════════════════
elif page == "histogram":
    st.markdown(
        '<div class="hero" style="padding:1.5rem 2rem">'
        '<h1 style="font-size:1.8rem">📊 Histogram — היסטוגרמה</h1>'
        '<p>התפלגות של משתנה רציף</p></div>',
        unsafe_allow_html=True,
    )

    when_box([
        "המשתנה הוא <b>רציף</b> (מספרים כמו: דופק, חום, גיל...)",
        "רוצים לראות: אילו ערכים שכיחים? מה הטווח? האם יש פיק אחד או שניים?",
        "שאלה לדוגמה: <b>מה ההתפלגות של דופק הלב בדאטה שלנו?</b>",
    ])

    # ── Example 1 ──────────────────────────────────────────
    section("📌 דוגמה 1 — היסטוגרמה פשוטה")

    explain("""
    כדי לבנות היסטוגרמה ב-Streamlit:<br>
    1. בוחרים עמודה רציפה מה-DataFrame<br>
    2. מחלקים את הערכים ל-<b>bins</b> — טווחי ערכים<br>
    3. סופרים כמה ערכים נפלו בכל bin<br>
    4. מציגים עם <code>st.bar_chart()</code>
    """)

    st.code("""\
# שלב 1: בחירת עמודה
col = df["heart_rate"]

# שלב 2+3: חלוקה ל-bins וספירה
counts = pd.cut(col, bins=10).value_counts().sort_index()

# שלב 4: הצגה
st.bar_chart(counts)
""", language="python")

    explain("""
    <b>pd.cut(col, bins=10)</b> — מחלקת את הערכים ל-10 טווחים שווים.<br>
    <b>.value_counts()</b> — סופרת כמה ערכים נפלו בכל טווח.<br>
    <b>.sort_index()</b> — ממיינת מקטן לגדול, כדי שהגרף יוצג בסדר נכון.
    """)

    param_table([
        ("pd.cut(col, bins=N)", "function", "מחלק משתנה רציף ל-N טווחים שווים"),
        (".value_counts()",     "method",   "סופרת כמה ערכים יש בכל טווח"),
        (".sort_index()",       "method",   "ממיין לפי סדר הטווחים (ציר X)"),
        ("st.bar_chart()",      "function", "מציגה את ספירת ה-bins כגרף עמודות"),
    ])

    counts = pd.cut(df["heart_rate"], bins=10).value_counts().sort_index()
    st.bar_chart(counts, use_container_width=True)
    st.caption("ציר X = טווחי ערכי דופק. ציר Y = כמה מטופלים בכל טווח.")

    divider()

    # ── Example 2 ──────────────────────────────────────────
    section("📌 דוגמה 2 — השוואה: תקין מול לא תקין")

    explain("""
    שאלה: <b>האם ההתפלגות של הדופק שונה בין מטופלים תקינים ללא תקינים?</b><br>
    כדי לענות — מחשבים היסטוגרמה נפרדת לכל קבוצה ומציגים יחד.
    """)

    st.code("""\
# מחלקים לשתי קבוצות
normal   = df[df["test_normal"] == 1]["heart_rate"]
abnormal = df[df["test_normal"] == 0]["heart_rate"]

# מגדירים bins זהים לשתי הקבוצות!
bins = range(40, 140, 10)

counts_normal   = pd.cut(normal,   bins=bins).value_counts().sort_index()
counts_abnormal = pd.cut(abnormal, bins=bins).value_counts().sort_index()

# מאחדים לטבלה אחת
hist_df = pd.DataFrame({
    "Normal (1)":   counts_normal,
    "Abnormal (0)": counts_abnormal,
})

st.bar_chart(hist_df)
""", language="python")

    explain("""
    <b>df[df["test_normal"] == 1]</b> — מסנן רק שורות שבהן test_normal שווה ל-1 (תקין).<br>
    <b>bins = range(40, 140, 10)</b> — bins קבועים משותפים לשתי הקבוצות. חשוב! כדי שהשוואה תהיה הוגנת.<br>
    <b>pd.DataFrame({...})</b> — מאחד שתי סדרות לטבלה אחת — Streamlit יציג אותן זו לצד זו.
    """)

    bins = range(40, 140, 10)
    cn = pd.cut(df[df["test_normal"]==1]["heart_rate"], bins=bins).value_counts().sort_index()
    ca = pd.cut(df[df["test_normal"]==0]["heart_rate"], bins=bins).value_counts().sort_index()
    hist_df = pd.DataFrame({"Normal (1)": cn, "Abnormal (0)": ca})
    st.bar_chart(hist_df, use_container_width=True)
    st.caption("כחול = תקין, כתום = לא תקין. לא תקינים מרוכזים בערכי דופק גבוהים יותר.")

    divider()

    # ── Interactive ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.selectbox(
            "בחר משתנה רציף:",
            ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"],
        )
        n_bins = st.slider("כמה bins?", 5, 20, 10)
        split  = st.checkbox("הצג לפי תקין / לא תקין")
    with col2:
        if not split:
            c = pd.cut(df[chosen], bins=n_bins).value_counts().sort_index()
            st.bar_chart(c, use_container_width=True)
        else:
            cn2 = pd.cut(df[df["test_normal"]==1][chosen], bins=n_bins).value_counts().sort_index()
            ca2 = pd.cut(df[df["test_normal"]==0][chosen], bins=n_bins).value_counts().sort_index()
            combined = pd.DataFrame({"Normal (1)": cn2, "Abnormal (0)": ca2}).fillna(0)
            st.bar_chart(combined, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# BAR CHART
# ═══════════════════════════════════════════════════════════
elif page == "bar_chart":
    st.markdown(
        '<div class="hero" style="padding:1.5rem 2rem">'
        '<h1 style="font-size:1.8rem">📊 Bar Chart — גרף עמודות</h1>'
        '<p>כמות תצפיות לפי קטגוריה</p></div>',
        unsafe_allow_html=True,
    )

    when_box([
        "המשתנה הוא <b>בדיד / קטגוריאלי</b> (כמו: gender, test_normal)",
        "רוצים לראות: <b>כמה תצפיות</b> יש בכל קטגוריה?",
        "שאלה לדוגמה: <b>כמה מטופלים תקינים וכמה לא תקינים יש בדאטה?</b>",
    ])

    # ── Example 1 ──────────────────────────────────────────
    section("📌 דוגמה 1 — ספירת קטגוריה אחת")

    explain("הצעד הכי פשוט: לספור כמה פעמים מופיע כל ערך בעמודה.")

    st.code("""\
# ספירת כמה תקינים ולא תקינים יש
counts = df["test_normal"].value_counts()

st.bar_chart(counts)
""", language="python")

    explain("""
    <b>df["test_normal"]</b> — בוחרים את העמודה הקטגוריאלית.<br>
    <b>.value_counts()</b> — סופרת כמה פעמים מופיע כל ערך (0 ו-1).<br>
    <b>st.bar_chart()</b> — עמודה אחת לכל קטגוריה, גובה העמודה = הכמות.
    """)

    param_table([
        ("df[\"col\"]",        "Series",   "עמודה מתוך ה-DataFrame"),
        (".value_counts()",    "method",   "מחזירה Series: קטגוריה → כמות"),
        ("st.bar_chart(data)", "function", "גרף עמודות — ציר X = קטגוריות, Y = כמות"),
    ])

    counts = df["test_normal"].value_counts()
    counts.index = counts.index.map({0: "Abnormal (0)", 1: "Normal (1)"})
    st.bar_chart(counts, use_container_width=True)
    st.caption("ציר X = קטגוריה. ציר Y = מספר מטופלים.")

    divider()

    # ── Example 2 ──────────────────────────────────────────
    section("📌 דוגמה 2 — ספירה לפי שתי עמודות")

    explain("""
    שאלה: <b>מתוך גברים ונשים — כמה מכל אחד תקינים ולא תקינים?</b><br>
    כלומר לספור לפי <b>שתי עמודות בו-זמנית</b>.
    """)

    st.code("""\
# ספירה לפי gender וגם לפי test_normal
counts = df.groupby("gender")["test_normal"].value_counts().unstack()

st.bar_chart(counts)
""", language="python")

    explain("""
    <b>df.groupby("gender")</b> — מחלקת את הדאטה לשתי קבוצות: gender=0 ו-gender=1.<br>
    <b>["test_normal"].value_counts()</b> — בכל קבוצה, סופרת כמה תקינים וכמה לא תקינים.<br>
    <b>.unstack()</b> — הופכת את התוצאה לטבלה: שורה = gender, עמודה = test_normal.<br>
    התוצאה: גרף עמודות מקובץ — עמודה לכל gender, צבע לכל תוצאת בדיקה.
    """)

    param_table([
        ("df.groupby(\"col\")",       "method", "מחלק את הדאטה לקבוצות לפי עמודה"),
        ("[\"col2\"].value_counts()", "method", "סופרת קטגוריות בתוך כל קבוצה"),
        (".unstack()",                "method", "הופכת multi-index לטבלה רחבה — עמודה לכל ערך"),
    ])

    counts2 = df.groupby("gender")["test_normal"].value_counts().unstack().fillna(0)
    counts2.index   = counts2.index.map({0: "Female (0)", 1: "Male (1)"})
    counts2.columns = counts2.columns.map({0: "Abnormal (0)", 1: "Normal (1)"})
    st.bar_chart(counts2, use_container_width=True)
    st.caption("כל קבוצת עמודות = gender. כל צבע = תוצאת בדיקה.")

    divider()

    # ── Interactive ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    col1, col2 = st.columns([1, 2])
    with col1:
        cat_col   = st.selectbox("בחר עמודה קטגוריאלית:", ["test_normal", "gender"])
        add_split = st.checkbox("פצל לפי עמודה שנייה")
        split_col = None
        if add_split:
            options   = [c for c in ["test_normal", "gender"] if c != cat_col]
            split_col = st.selectbox("פצל לפי:", options)
    with col2:
        if not add_split or split_col is None:
            c = df[cat_col].value_counts()
            c.index = c.index.astype(str)
            st.bar_chart(c, use_container_width=True)
        else:
            c = df.groupby(cat_col)[split_col].value_counts().unstack().fillna(0)
            c.index   = c.index.astype(str)
            c.columns = c.columns.astype(str)
            st.bar_chart(c, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SCATTER PLOT
# ═══════════════════════════════════════════════════════════
elif page == "scatter":
    st.markdown(
        '<div class="hero" style="padding:1.5rem 2rem">'
        '<h1 style="font-size:1.8rem">🔵 Scatter Plot — גרף פיזור</h1>'
        '<p>קשר בין שני משתנים רציפים</p></div>',
        unsafe_allow_html=True,
    )

    when_box([
        "שני המשתנים הם <b>רציפים</b> (מספרים)",
        "רוצים לראות: האם כשאחד עולה, השני עולה גם? (קורלציה)",
        "שאלה לדוגמה: <b>האם מי שיש לו דופק גבוה גם סובל מחום גבוה?</b>",
    ])

    # ── Example 1 ──────────────────────────────────────────
    section("📌 דוגמה 1 — גרף פיזור בסיסי")

    explain("""
    הגרף הפשוט ביותר: כל נקודה = מטופל אחד.<br>
    ציר X = ערך משתנה אחד. ציר Y = ערך משתנה שני.
    """)

    st.code("""\
st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
)
""", language="python")

    explain("""
    <b>st.scatter_chart(df)</b> — מקבל את ה-DataFrame ויוצר גרף פיזור.<br>
    <b>x="heart_rate"</b> — שם העמודה לציר האופקי (ציר X).<br>
    <b>y="body_temp"</b> — שם העמודה לציר האנכי (ציר Y).<br>
    כל נקודה בגרף = שורה אחת ב-DataFrame = מטופל אחד.
    """)

    param_table([
        ("st.scatter_chart(df)", "function", "גרף פיזור מובנה — מקבל DataFrame"),
        ("x",                    "str",      "שם עמודת ציר X — חובה"),
        ("y",                    "str",      "שם עמודת ציר Y — חובה"),
        ("color",                "str",      "שם עמודה לצביעת נקודות — אופציונלי"),
        ("size",                 "str",      "שם עמודה לגודל נקודות — אופציונלי"),
    ])

    st.scatter_chart(df, x="heart_rate", y="body_temp", use_container_width=True)
    st.caption("כל נקודה = מטופל. X = דופק, Y = חום. כרגע כל הנקודות בצבע אחד.")

    divider()

    # ── Example 2 ──────────────────────────────────────────
    section("📌 דוגמה 2 — הוספת color לפי קטגוריה")

    explain("""
    הוספת <b>color</b> היא הדרך החזקה ביותר לגרף פיזור —
    פתאום אפשר לראות האם שתי קבוצות (תקין / לא תקין) נמצאות באזורים שונים בגרף.
    """)

    st.code("""\
# חשוב: ממירים ל-str כדי לקבל צבעים בדידים
df["test_normal"] = df["test_normal"].astype(str)

st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
    color="test_normal",   # <-- שורה אחת בלבד!
)
""", language="python")

    explain("""
    <b>color="test_normal"</b> — Streamlit מצבע כל נקודה לפי ערך העמודה.<br>
    <b>.astype(str)</b> — חשוב! בלי זה Streamlit ייצג צבע כסקאלה רציפה (כחול-לבן) במקום שני צבעים בדידים.<br>
    כחול = ערך "1" (תקין), כתום = ערך "0" (לא תקין).
    """)

    df_plot = df.copy()
    df_plot["test_normal"] = df_plot["test_normal"].astype(str)
    st.scatter_chart(df_plot, x="heart_rate", y="body_temp", color="test_normal", use_container_width=True)
    st.caption("כחול = תקין (1), כתום = לא תקין (0). רואים שהקבוצות מופרדות!")

    divider()

    # ── Example 3 ──────────────────────────────────────────
    section("📌 דוגמה 3 — הוספת size לפי משתנה שלישי")

    explain("""
    אפשר להוסיף <b>ממד שלישי</b>: גודל הנקודה מייצג משתנה נוסף.<br>
    נקודה גדולה = ערך גבוה, נקודה קטנה = ערך נמוך.
    """)

    st.code("""\
# יוצרים עמודה עם ערכים גדולים מספיק לגודל הנקודה
df["protein_size"] = df["protein_level"] * 5

st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
    color="test_normal",
    size="protein_size",   # <-- גודל לפי חלבון
)
""", language="python")

    explain("""
    <b>size="protein_size"</b> — עמודה מספרית שתקבע את גודל כל נקודה.<br>
    כפלנו ב-5 כי Streamlit צריך ערכים גדולים מספיק כדי שהגדלים יהיו נראים.<br>
    ככה: נקודה גדולה = רמת חלבון גבוהה, קטנה = נמוכה.
    """)

    df_plot2 = df.copy()
    df_plot2["test_normal"]  = df_plot2["test_normal"].astype(str)
    df_plot2["protein_size"] = df_plot2["protein_level"] * 5
    st.scatter_chart(df_plot2, x="heart_rate", y="body_temp", color="test_normal", size="protein_size", use_container_width=True)
    st.caption("גודל הנקודה = רמת חלבון. נקודה גדולה = חלבון גבוה.")

    divider()

    # ── Interactive ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    cont_cols = ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age"]
    col1, col2 = st.columns([1, 2])
    with col1:
        x_col     = st.selectbox("ציר X:", cont_cols, index=0)
        y_col     = st.selectbox("ציר Y:", cont_cols, index=1)
        use_color = st.checkbox("צבע לפי test_normal", value=True)
        use_size  = st.checkbox("גודל לפי משתנה נוסף")
        size_col  = None
        if use_size:
            remaining = [c for c in cont_cols if c not in [x_col, y_col]]
            if remaining:
                size_col = st.selectbox("גודל לפי:", remaining)
    with col2:
        cols_needed = list({x_col, y_col, "test_normal"})
        if size_col:
            cols_needed.append(size_col)
        df_i = df[cols_needed].copy()
        if use_color:
            df_i["test_normal"] = df_i["test_normal"].astype(str)
        kwargs = {"x": x_col, "y": y_col, "use_container_width": True}
        if use_color:
            kwargs["color"] = "test_normal"
        if use_size and size_col:
            scale = 5 if df_i[size_col].mean() < 100 else 0.05
            df_i["_size"] = df_i[size_col] * scale
            kwargs["size"] = "_size"
        st.scatter_chart(df_i, **kwargs)
