import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="מדריך ויזואליזציה | Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Assistant', sans-serif; direction: rtl; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; direction: rtl; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1000px; }
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-radius: 14px; padding: 2rem; margin-bottom: 2rem;
    color: white; direction: rtl;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem; }
.hero p  { font-size: 1rem; color: #94a3b8; margin: 0; }
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.3rem; margin: 1.8rem 0 1rem; direction: rtl;
}
.when-box {
    background: #eff6ff; border-right: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem; direction: rtl;
}
.when-box strong { color: #1d4ed8; }
.when-box ul { margin: 0.4rem 0 0; padding-right: 1.2rem; color: #334155; }
.when-box li { margin-bottom: 0.2rem; font-size: 0.95rem; }
.explain-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem; direction: rtl; line-height: 1.8;
}
.explain-box b { color: #1e40af; }
.stCodeBlock, code, pre { direction: ltr !important; text-align: left !important; }
</style>
""", unsafe_allow_html=True)


# ── Dataset ─────────────────────────────────────────────────
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
# עמודת label קריאה לצבעים בדידים
df["label"] = df["test_normal"].map({1: "Normal", 0: "Abnormal"})


# ── Sidebar ──────────────────────────────────────────────────
PAGES = {
    "🏠 דף הבית":      "home",
    "📊 Histogram":    "histogram",
    "📊 Bar Chart":    "bar_chart",
    "🔵 Scatter Plot": "scatter",
}
with st.sidebar:
    st.markdown("## 📊 מדריך ויזואליזציה")
    st.markdown("---")
    page_label = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.82rem;color:#94a3b8;line-height:1.8'>"
        "דאטהסט: בדיקות דם<br>100 תצפיות · 8 עמודות<br>"
        "מטרה: <code>test_normal</code></div>",
        unsafe_allow_html=True,
    )


# ── Helpers ──────────────────────────────────────────────────
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


# ════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class="hero">
      <h1>📊 מדריך ויזואליזציה ב-Streamlit</h1>
      <p>הסברים פשוטים לבניית גרפים — עם דוגמאות קוד וגרפים חיים</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🩺 הדאטהסט")
    explain("""
    הדאטהסט מכיל <b>100 רשומות של בדיקות דם</b>. כל שורה = מטופל אחד.<br><br>
    <b>משתנים רציפים:</b> heart_rate · body_temp · protein_level · white_blood_cells · hemoglobin · age<br>
    <b>משתנים בדידים:</b> gender (0/1) · <b>test_normal — 1=תקין, 0=לא תקין ← משתנה המטרה!</b>
    """)
    st.dataframe(df.drop(columns=["label"]).head(8), use_container_width=True)


# ════════════════════════════════════════════════════════════
# HISTOGRAM
# ════════════════════════════════════════════════════════════
elif page == "histogram":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem"><h1 style="font-size:1.8rem">📊 Histogram — היסטוגרמה</h1><p>התפלגות של משתנה רציף</p></div>', unsafe_allow_html=True)

    when_box([
        "המשתנה הוא <b>רציף</b> — מספרים כמו דופק, חום, גיל",
        "רוצים לראות: אילו ערכים שכיחים? מה הטווח? האם יש פיק אחד או שניים?",
        "שאלה לדוגמה: <b>מה ההתפלגות של דופק הלב בדאטה?</b>",
    ])

    # ── דוגמה 1 ────────────────────────────────────────────
    section("📌 דוגמה 1 — היסטוגרמה פשוטה")
    explain("""
    ב-Streamlit אין פונקציית histogram מובנית, אז עושים זאת בשני שלבים:<br>
    1. מחלקים את הערכים לטווחים (<b>bins</b>) ומחשבים כמות בכל טווח — עם pandas<br>
    2. מציגים את הכמויות כגרף עמודות עם <code>st.bar_chart()</code>
    """)

    st.code("""\
# שלב 1: חלוקה ל-bins וספירה
counts = pd.cut(df["heart_rate"], bins=10).value_counts().sort_index()

# שלב 2: הצגה
st.bar_chart(counts)
""", language="python")

    explain("""
    <b>pd.cut(df["heart_rate"], bins=10)</b> — מחלקת את הערכים ל-10 טווחים שווים.<br>
    <b>.value_counts()</b> — סופרת כמה ערכים בכל טווח.<br>
    <b>.sort_index()</b> — מסדרת מקטן לגדול, כך שהגרף מוצג בסדר נכון.
    """)

    counts = pd.cut(df["heart_rate"], bins=10).value_counts().sort_index()
    st.bar_chart(counts)
    st.caption("ציר X = טווחי ערכי דופק. ציר Y = כמה מטופלים בכל טווח.")

    st.divider()

    # ── דוגמה 2 ────────────────────────────────────────────
    section("📌 דוגמה 2 — השוואה: תקין מול לא תקין")
    explain("""
    שאלה: <b>האם ההתפלגות של הדופק שונה בין תקינים ללא תקינים?</b><br>
    פותרים: מחלקים לשתי קבוצות, מחשבים histogrm לכל אחת, ומציגים יחד.
    """)

    st.code("""\
# מסננים כל קבוצה בנפרד
normal   = df[df["test_normal"] == 1]["heart_rate"]
abnormal = df[df["test_normal"] == 0]["heart_rate"]

# bins זהים לשתי הקבוצות — חשוב להשוואה הוגנת!
bins = range(40, 140, 10)

counts_normal   = pd.cut(normal,   bins=bins).value_counts().sort_index()
counts_abnormal = pd.cut(abnormal, bins=bins).value_counts().sort_index()

# מאחדים DataFrame אחד — Streamlit יצבע כל עמודה בצבע אחר
hist_df = pd.DataFrame({
    "Normal":   counts_normal,
    "Abnormal": counts_abnormal,
})

st.bar_chart(hist_df)
""", language="python")

    explain("""
    <b>df[df["test_normal"] == 1]</b> — מסנן רק שורות שבהן test_normal שווה ל-1.<br>
    <b>bins = range(40, 140, 10)</b> — טווחים קבועים משותפים לשתי הקבוצות. בלי זה הגרף לא מדויק.<br>
    <b>pd.DataFrame({"Normal": ..., "Abnormal": ...})</b> — כשמעבירים DataFrame עם שתי עמודות,
    Streamlit מציג כל עמודה בצבע שונה אוטומטית.
    """)

    bins = range(40, 140, 10)
    cn = pd.cut(df[df["test_normal"]==1]["heart_rate"], bins=bins).value_counts().sort_index()
    ca = pd.cut(df[df["test_normal"]==0]["heart_rate"], bins=bins).value_counts().sort_index()
    st.bar_chart(pd.DataFrame({"Normal": cn, "Abnormal": ca}))
    st.caption("כחול = תקין, כתום = לא תקין. לא תקינים מרוכזים בדופק גבוה יותר.")

    st.divider()

    # ── נסו בעצמכם ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.selectbox("בחר משתנה רציף:",
            ["heart_rate","body_temp","protein_level","white_blood_cells","hemoglobin","age"])
        n_bins = st.slider("כמה bins?", 5, 20, 10)
        split  = st.checkbox("הצג לפי תקין / לא תקין")
    with col2:
        if not split:
            st.bar_chart(pd.cut(df[chosen], bins=n_bins).value_counts().sort_index())
        else:
            b = n_bins
            cn = pd.cut(df[df["test_normal"]==1][chosen], bins=b).value_counts().sort_index()
            ca = pd.cut(df[df["test_normal"]==0][chosen], bins=b).value_counts().sort_index()
            st.bar_chart(pd.DataFrame({"Normal": cn, "Abnormal": ca}).fillna(0))


# ════════════════════════════════════════════════════════════
# BAR CHART
# ════════════════════════════════════════════════════════════
elif page == "bar_chart":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem"><h1 style="font-size:1.8rem">📊 Bar Chart — גרף עמודות</h1><p>כמות תצפיות לפי קטגוריה</p></div>', unsafe_allow_html=True)

    when_box([
        "המשתנה הוא <b>בדיד / קטגוריאלי</b> — כמו gender, test_normal",
        "רוצים לראות: <b>כמה תצפיות</b> יש בכל קטגוריה?",
        "שאלה לדוגמה: <b>כמה מטופלים תקינים וכמה לא תקינים?</b>",
    ])

    # ── דוגמה 1 ────────────────────────────────────────────
    section("📌 דוגמה 1 — ספירת קטגוריה אחת")
    explain("הצעד הכי פשוט: לספור כמה פעמים מופיע כל ערך בעמודה.")

    st.code("""\
# ספירת כמה תקינים ולא תקינים יש
counts = df["test_normal"].value_counts()

st.bar_chart(counts)
""", language="python")

    explain("""
    <b>df["test_normal"].value_counts()</b> — מחזירה Series:
    כמה פעמים מופיע כל ערך (0 ו-1).<br>
    <b>st.bar_chart(counts)</b> — ה-index הופך לציר X, הערכים לציר Y.
    """)

    st.bar_chart(df["test_normal"].value_counts())
    st.caption("ציר X = קטגוריה (0/1). ציר Y = כמות מטופלים.")

    st.divider()

    # ── דוגמה 2 ────────────────────────────────────────────
    section("📌 דוגמה 2 — ספירה עם color לפי קטגוריה שנייה")
    explain("""
    שאלה: <b>כמה תקינים ולא תקינים יש — בנפרד לגברים ונשים?</b><br>
    הפתרון: להשתמש ב-<code>x</code>, <code>y</code>, ו-<code>color</code> של <code>st.bar_chart</code>.
    """)

    st.code("""\
# מכינים DataFrame עם ספירה לכל שילוב
counts = (
    df.groupby(["gender", "test_normal"])
      .size()
      .reset_index(name="count")
)
# gender ו-test_normal הופכים לעמודות רגילות

st.bar_chart(counts, x="gender", y="count", color="test_normal")
""", language="python")

    explain("""
    <b>df.groupby(["gender","test_normal"]).size()</b> — סופרת כמה שורות יש לכל שילוב של gender + test_normal.<br>
    <b>.reset_index(name="count")</b> — הופכת את התוצאה ל-DataFrame עם עמודות רגילות.<br>
    <b>x="gender"</b> — ציר X = קטגוריית gender.<br>
    <b>y="count"</b> — ציר Y = הכמות.<br>
    <b>color="test_normal"</b> — כל ערך של test_normal מקבל צבע שונה.
    """)

    counts2 = (
        df.groupby(["gender", "test_normal"])
          .size()
          .reset_index(name="count")
    )
    counts2["test_normal"] = counts2["test_normal"].astype(str)
    counts2["gender"]      = counts2["gender"].astype(str)
    st.bar_chart(counts2, x="gender", y="count", color="test_normal")
    st.caption("כל עמודה = gender. כל צבע = תוצאת בדיקה. רואים שהחלוקה דומה בין גברים לנשים.")

    st.divider()

    # ── נסו בעצמכם ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    col1, col2 = st.columns([1, 2])
    with col1:
        x_col  = st.selectbox("ציר X (קטגוריה):", ["gender", "test_normal"])
        color_col = st.selectbox("צבע לפי:", ["test_normal", "gender"])
    with col2:
        c = (
            df.groupby([x_col, color_col])
              .size()
              .reset_index(name="count")
        )
        c[x_col]      = c[x_col].astype(str)
        c[color_col]  = c[color_col].astype(str)
        st.bar_chart(c, x=x_col, y="count", color=color_col)


# ════════════════════════════════════════════════════════════
# SCATTER PLOT
# ════════════════════════════════════════════════════════════
elif page == "scatter":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem"><h1 style="font-size:1.8rem">🔵 Scatter Plot — גרף פיזור</h1><p>קשר בין שני משתנים רציפים</p></div>', unsafe_allow_html=True)

    when_box([
        "שני המשתנים הם <b>רציפים</b> — מספרים",
        "רוצים לראות: האם כשאחד עולה, השני עולה גם?",
        "שאלה לדוגמה: <b>האם מי שיש לו דופק גבוה גם סובל מחום גבוה?</b>",
    ])

    # ── דוגמה 1 ────────────────────────────────────────────
    section("📌 דוגמה 1 — גרף פיזור בסיסי: x ו-y")
    explain("""
    הגרף הפשוט ביותר: כל נקודה = מטופל אחד.<br>
    מגדירים: <b>x</b> = שם עמודה לציר האופקי, <b>y</b> = שם עמודה לציר האנכי.
    """)

    st.code("""\
st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
)
""", language="python")

    explain("""
    <b>st.scatter_chart(df, ...)</b> — מקבל DataFrame ויוצר גרף פיזור.<br>
    <b>x="heart_rate"</b> — שם העמודה לציר X.<br>
    <b>y="body_temp"</b> — שם העמודה לציר Y.<br>
    כל נקודה = שורה אחת ב-DataFrame = מטופל אחד.
    """)

    st.scatter_chart(df, x="heart_rate", y="body_temp")
    st.caption("כל נקודה = מטופל. X = דופק, Y = חום. כל הנקודות בצבע אחד.")

    st.divider()

    # ── דוגמה 2 ────────────────────────────────────────────
    section("📌 דוגמה 2 — הוספת color לפי קטגוריה")
    explain("""
    עכשיו נוסיף <b>color</b> — Streamlit יצבע כל נקודה לפי ערך של עמודה אחרת.<br>
    ככה נראה בבירור האם תקינים ולא תקינים נמצאים באזורים שונים בגרף.
    """)

    st.code("""\
st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
    color="label",     # עמודת טקסט: "Normal" / "Abnormal"
)
""", language="python")

    explain("""
    <b>color="label"</b> — שם עמודה שמכילה טקסט (כמו "Normal", "Abnormal").<br>
    Streamlit בוחר צבע אחר לכל ערך ייחודי אוטומטית.<br>
    <b>חשוב:</b> אם העמודה מכילה מספרים (0/1), Streamlit ישתמש בסקאלת צבעים רציפה —
    לא מה שרוצים. לכן עדיף להשתמש בעמודת טקסט.
    """)

    st.scatter_chart(df, x="heart_rate", y="body_temp", color="label")
    st.caption("כל צבע = קבוצה (Normal / Abnormal). רואים הפרדה ברורה!")

    st.divider()

    # ── דוגמה 3 ────────────────────────────────────────────
    section("📌 דוגמה 3 — הוספת size לפי משתנה שלישי")
    explain("""
    אפשר להוסיף ממד נוסף: <b>size</b> — גודל הנקודה לפי משתנה מספרי.<br>
    נקודה גדולה = ערך גבוה. נקודה קטנה = ערך נמוך.
    """)

    st.code("""\
st.scatter_chart(
    df,
    x="heart_rate",
    y="body_temp",
    color="label",
    size="hemoglobin",   # גודל לפי רמת המוגלובין
)
""", language="python")

    explain("""
    <b>size="hemoglobin"</b> — שם עמודה מספרית שתקבע את גודל כל נקודה.<br>
    Streamlit מתרגם את הערכים לגדלים אוטומטית — אין צורך להכפיל ב-5 או בכלום.
    """)

    st.scatter_chart(df, x="heart_rate", y="body_temp", color="label", size="hemoglobin")
    st.caption("גודל הנקודה = רמת המוגלובין.")

    st.divider()

    # ── נסו בעצמכם ─────────────────────────────────────────
    section("🎛️ נסו בעצמכם")
    cont_cols = ["heart_rate","body_temp","protein_level","white_blood_cells","hemoglobin","age"]
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
        kwargs = {"x": x_col, "y": y_col}
        if use_color:
            kwargs["color"] = "label"
        if use_size and size_col:
            kwargs["size"] = size_col
        st.scatter_chart(df, **kwargs)
