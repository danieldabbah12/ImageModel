import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="מדריך גרפים ב-Streamlit",
    layout="wide",
)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
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
        "test_normal":       1,
    })
    abnormal = pd.DataFrame({
        "heart_rate":        np.random.normal(90, 15, half),
        "body_temp":         np.random.normal(37.8, 0.8, half),
        "protein_level":     np.random.normal(6.0, 1.0, half),
        "white_blood_cells": np.random.normal(11000, 3000, half),
        "hemoglobin":        np.random.normal(11.5, 2.0, half),
        "age":               np.random.randint(18, 80, half),
        "gender":            np.random.randint(0, 2, half),
        "test_normal":       0,
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

# ──────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { direction: rtl; text-align: right; }

    .tutorial-box {
        background: #f0f4ff;
        border-right: 5px solid #1a56db;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.7;
    }
    .command-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #1a56db;
        margin-bottom: 0.3rem;
    }
    .tip-box {
        background: #f0fdf4;
        border-right: 4px solid #16a34a;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #14532d;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fff7ed;
        border-right: 4px solid #ea580c;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #7c2d12;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .step {
        background: #1e293b;
        color: #94a3b8;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## נושאים")
    section = st.radio("בחר נושא:", [
        "הכרת הנתונים",
        "נתון בדיד vs. רציף",
        "st.bar_chart",
        "st.line_chart",
        "st.scatter_chart",
        "Histogram",
    ])
    st.markdown("---")
    st.markdown("**המטרה:** לחזות את `test_normal`")
    st.markdown("`1` = תקין | `0` = לא תקין")


# ══════════════════════════════════════════════
# SECTION 1 – הכרת הנתונים
# ══════════════════════════════════════════════
if section == "הכרת הנתונים":

    st.title("הכרת הנתונים")
    st.markdown("לפני שמציירים גרפים — צריך להבין מה יש לנו. נתחיל בשתי פקודות בסיסיות.")

    st.markdown("---")

    # st.dataframe
    st.markdown('<div class="command-title">הפקודה: st.dataframe()</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="tutorial-box">
הפקודה <code>st.dataframe()</code> מציגה טבלה אינטראקטיבית של ה-DataFrame שלנו.
זו הדרך הכי מהירה לראות איך הנתונים נראים.
</div>
""", unsafe_allow_html=True)

    st.markdown("**הקוד:**")
    st.code("st.dataframe(df.head())", language="python")

    st.markdown("**התוצאה:**")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # st.metric
    st.markdown('<div class="command-title">הפקודה: st.metric()</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="tutorial-box">
הפקודה <code>st.metric()</code> מציגה מספר בודד בצורה בולטת.
שימושית להצגת סטטיסטיקות מפתח על הנתונים.
</div>
""", unsafe_allow_html=True)

    st.markdown("**הקוד:**")
    st.code("""col1, col2, col3 = st.columns(3)
col1.metric("Patients",  len(df))
col2.metric("Normal",    int(df["test_normal"].sum()))
col3.metric("Abnormal",  int((df["test_normal"] == 0).sum()))
""", language="python")

    st.markdown("**התוצאה:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients",  len(df))
    col2.metric("Normal",    int(df["test_normal"].sum()))
    col3.metric("Abnormal",  int((df["test_normal"] == 0).sum()))


# ══════════════════════════════════════════════
# SECTION 2 – נתון בדיד vs. רציף
# ══════════════════════════════════════════════
elif section == "נתון בדיד vs. רציף":

    st.title("נתון בדיד לעומת נתון רציף")
    st.markdown("לפני שבוחרים גרף — צריך להבין מה סוג הנתון. זה קובע איזה גרף להשתמש.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### נתון בדיד (Discrete)")
        st.markdown("""
<div class="tutorial-box">
נתון שיכול לקבל <strong>מספר קטן של ערכים קבועים</strong>.

דוגמאות בנתונים שלנו:
- <code>test_normal</code> — רק 0 או 1
- <code>gender</code> — רק 0 או 1
- <code>age</code> — מספרים שלמים בלבד

השאלה שאנחנו שואלים: <em>"כמה מכל קטגוריה?"</em>

הגרף המתאים: Bar Chart
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### נתון רציף (Continuous)")
        st.markdown("""
<div class="tutorial-box">
נתון שיכול לקבל <strong>כל ערך</strong> בטווח מסוים.

דוגמאות בנתונים שלנו:
- <code>heart_rate</code> — 60.0, 72.5, 95.3...
- <code>body_temp</code> — 36.2, 37.1, 38.4...
- <code>hemoglobin</code> — 11.5, 14.2...

השאלה שאנחנו שואלים: <em>"מה הטווח? יש הבדל בין קבוצות?"</em>

הגרפים המתאימים: Histogram, Scatter
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div class="tip-box">
כלל אצבע: אם אפשר לספור את כל הערכים האפשריים על אצבעות — זה כנראה בדיד.
אם יש הרבה ערכים שונים (כמו דופק) — זה רציף.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 3 – Bar Chart
# ══════════════════════════════════════════════
elif section == "st.bar_chart":

    st.title("st.bar_chart — גרף עמודות")

    st.markdown("""
<div class="tutorial-box">
<strong>מתי משתמשים?</strong> כשרוצים לראות <strong>כמה רשומות יש בכל קטגוריה</strong>.
מתאים לנתון בדיד.<br><br>
<strong>התחביר הבסיסי:</strong><br>
<code>st.bar_chart(data)</code><br><br>
הפונקציה מצפה לקבל Series או DataFrame שבו האינדקס הוא ציר X, והעמודות הן ציר Y.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### דוגמה — כמה מטופלים תקינים ולא תקינים?")
    st.markdown("ניצור Series שסופרת כמה פעמים מופיע כל ערך, ונציג אותה:")

    st.code("""counts = df["test_normal"].value_counts()
st.bar_chart(counts)
""", language="python")

    st.markdown("**התוצאה:**")
    counts = df["test_normal"].value_counts()
    st.bar_chart(counts)

    st.markdown("""
<div class="tip-box">
מה לחפש: האם שני העמודות בגובה דומה? אם לא — הנתונים לא מאוזנים, וזה משפיע על המודל.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 4 – Line Chart
# ══════════════════════════════════════════════
elif section == "st.line_chart":

    st.title("st.line_chart — גרף קווים")

    st.markdown("""
<div class="tutorial-box">
<strong>מתי משתמשים?</strong> כשרוצים לראות <strong>שינוי לאורך ציר מסודר</strong> — כמו גיל או זמן.
השתמשו בקו כשיש משמעות לסדר של ציר X.<br><br>
<strong>התחביר הבסיסי:</strong><br>
<code>st.line_chart(data)</code><br><br>
מצפה ל-DataFrame שבו האינדקס הוא ציר X, והעמודות הן הקווים.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### דוגמה — heart_rate לפי גיל")
    st.markdown("נמיין לפי גיל, ונגדיר אותו כאינדקס:")

    st.code("""df_sorted = df.sort_values("age")
st.line_chart(df_sorted[["age", "heart_rate"]].set_index("age"))
""", language="python")

    st.markdown("**התוצאה:**")
    df_sorted = df.sort_values("age")
    st.line_chart(df_sorted[["age", "heart_rate"]].set_index("age"))

    st.markdown("""
<div class="warning-box">
שימו לב: כאן יש כמה מטופלים לאותו גיל, אז הקו יכול להיראות "רועש".
גרף קווים מתאים יותר כשלכל ערך X יש ערך Y אחד בלבד.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="tip-box">
אם ציר X הוא קטגוריות בלי סדר טבעי — עדיף להשתמש ב-Bar Chart.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 5 – Scatter Chart
# ══════════════════════════════════════════════
elif section == "st.scatter_chart":

    st.title("st.scatter_chart — גרף פיזור")

    st.markdown("""
<div class="tutorial-box">
<strong>מתי משתמשים?</strong> כשרוצים לראות את <strong>הקשר בין שני משתנים רציפים</strong>.
כל נקודה היא מטופל אחד.<br><br>
<strong>הטריק החשוב:</strong> אם נפצל את הנתונים לשתי עמודות — אחת לתקינים ואחת ללא תקינים —
Streamlit יצבע אותן בצבעים שונים. כך נוכל לראות אם הקבוצות נפרדות.<br><br>
<strong>התחביר הבסיסי:</strong><br>
<code>st.scatter_chart(data)</code>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("ציר X:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])
    with col2:
        y_col = st.selectbox("ציר Y:", ["hemoglobin", "heart_rate", "body_temp", "protein_level", "white_blood_cells"])

    st.markdown(f"### דוגמה — {x_col} מול {y_col}, מחולק לפי test_normal")
    st.markdown("נשים כל קבוצה בעמודה נפרדת. ה-`.where()` שם `NaN` בשורות שלא שייכות לקבוצה:")

    st.code(f"""plot_df = pd.DataFrame({{
    "x":        df["{x_col}"],
    "normal":   df["{y_col}"].where(df["test_normal"] == 1),
    "abnormal": df["{y_col}"].where(df["test_normal"] == 0),
}}).set_index("x")

st.scatter_chart(plot_df)
""", language="python")

    st.markdown("**התוצאה:**")
    plot_df = pd.DataFrame({
        "x":        df[x_col],
        "normal":   df[y_col].where(df["test_normal"] == 1),
        "abnormal": df[y_col].where(df["test_normal"] == 0),
    }).set_index("x")
    st.scatter_chart(plot_df)

    st.markdown("""
<div class="tip-box">
מה לחפש: אם שני הצבעים מתכנסים לאזורים שונים בגרף — המשתנים האלו הם מנבאים טובים של test_normal.
נסו להחליף בין השדות ולמצוא את הצמד שמפריד הכי טוב.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 6 – Histogram
# ══════════════════════════════════════════════
elif section == "Histogram":

    st.title("Histogram — גרף התפלגות")

    st.markdown("""
<div class="tutorial-box">
<strong>מתי משתמשים?</strong> כשרוצים לראות <strong>את ההתפלגות של משתנה רציף</strong> —
כלומר, באילו ערכים הוא מופיע הכי הרבה?<br><br>
<strong>איך עובד Histogram?</strong> מחלקים את הטווח ל"סלים" (bins), וסופרים כמה ערכים נפלו בכל סל.<br><br>
<strong>ב-Streamlit אין פקודת histogram ישירה</strong>, אז נחשב את הספירות עם NumPy ונציג עם <code>st.bar_chart()</code>.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    feature = st.selectbox("בחר שדה:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])
    bins    = st.slider("כמות bins:", 5, 30, 15)

    st.markdown(f"### דוגמה — התפלגות {feature} לפי test_normal")
    st.markdown("נחשב bins זהים לשתי הקבוצות, ונציג אותן זו לצד זו:")

    st.code(f"""edges = np.linspace(df["{feature}"].min(), df["{feature}"].max(), {bins} + 1)

normal_counts,   _ = np.histogram(df[df["test_normal"] == 1]["{feature}"], bins=edges)
abnormal_counts, _ = np.histogram(df[df["test_normal"] == 0]["{feature}"], bins=edges)

hist_df = pd.DataFrame(
    {{"normal": normal_counts, "abnormal": abnormal_counts}},
    index=[f"{{e:.1f}}" for e in edges[:-1]]
)
st.bar_chart(hist_df)
""", language="python")

    st.markdown("**התוצאה:**")
    edges = np.linspace(df[feature].min(), df[feature].max(), bins + 1)
    normal_counts,   _ = np.histogram(df[df["test_normal"] == 1][feature], bins=edges)
    abnormal_counts, _ = np.histogram(df[df["test_normal"] == 0][feature], bins=edges)
    hist_df = pd.DataFrame(
        {"normal": normal_counts, "abnormal": abnormal_counts},
        index=[f"{e:.1f}" for e in edges[:-1]]
    )
    st.bar_chart(hist_df)

    st.markdown("""
<div class="tip-box">
מה לחפש: אם שני הצבעים נמצאים באזורים שונים — השדה הזה מפריד טוב בין תקין ולא תקין.
אם הם חופפים הרבה — הוא פחות שימושי לחיזוי.
נסו להחליף שדות ולמצוא מי מפריד הכי טוב.
</div>
""", unsafe_allow_html=True)
