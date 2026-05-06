import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="מדריך ויזואליזציה ב-Streamlit",
    page_icon="📊",
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

    noise_size = int(n * noise_ratio)
    noise_idx  = np.random.choice(n, noise_size, replace=False)
    df.loc[noise_idx, "test_normal"] = 1 - df.loc[noise_idx, "test_normal"]

    df["heart_rate"]        = df["heart_rate"].round(0)
    df["body_temp"]         = df["body_temp"].round(1)
    df["protein_level"]     = df["protein_level"].round(2)
    df["white_blood_cells"] = df["white_blood_cells"].round(0)
    df["hemoglobin"]        = df["hemoglobin"].round(1)
    return df

df = generate_blood_test_dataset()

LABEL_MAP   = {1: "תקין", 0: "לא תקין"}
df["label"] = df["test_normal"].map(LABEL_MAP)

# ──────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { direction: rtl; text-align: right; }
    .big-title   { font-size: 2.4rem; font-weight: 800; color: #1a56db; margin-bottom: 0.2rem; }
    .sub-title   { font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem; }
    .section-box {
        background: #f0f4ff;
        border-right: 5px solid #1a56db;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }
    .code-explain {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: monospace;
        font-size: 0.88rem;
        margin-bottom: 1rem;
        white-space: pre-wrap;
    }
    .tag-discrete {
        background: #fef3c7; color: #92400e;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.8rem; font-weight: 700;
    }
    .tag-continuous {
        background: #dbeafe; color: #1e40af;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.8rem; font-weight: 700;
    }
    .tip-box {
        background: #ecfdf5; border-right: 4px solid #10b981;
        border-radius: 6px; padding: 0.7rem 1rem;
        color: #065f46; margin-top: 0.5rem;
    }
    hr { border: none; border-top: 2px solid #e5e7eb; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<div class="big-title">📊 מדריך ויזואליזציה ב-Streamlit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">מדריך מעשי לבניית גרפים — מבוסס על נתוני בדיקות דם</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR – navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂 ניווט")
    section = st.radio("בחר נושא:", [
        "1️⃣  הכרת ה-Dataset",
        "2️⃣  נתון בדיד vs. רציף",
        "3️⃣  Bar Chart — עמודות",
        "4️⃣  Line Chart — קווים",
        "5️⃣  Scatter Chart — פיזור",
        "6️⃣  Histogram — התפלגות",
        "7️⃣  Box Plot — תיבה",
    ])
    st.markdown("---")
    st.info("🎯 **מטרה:** לחזות את השדה `test_normal`\n\n`1` = תקין | `0` = לא תקין")

# ══════════════════════════════════════════════
# SECTION 1 – Dataset overview
# ══════════════════════════════════════════════
if section == "1️⃣  הכרת ה-Dataset":
    st.header("🔬 הכרת ה-Dataset")

    st.markdown("""
<div class="section-box">
לפני שמציירים גרפים — חשוב להבין מה יש לנו!<br>
ה-Dataset מכיל <strong>100 שורות</strong>, כל שורה היא מטופל אחד.
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👤 מטופלים",       len(df))
    col2.metric("📋 עמודות",        len(df.columns))
    col3.metric("✅ תקינים",         int(df["test_normal"].sum()))
    col4.metric("❌ לא תקינים",     int((df["test_normal"] == 0).sum()))

    st.markdown("### 👀 5 שורות ראשונות")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📑 תיאור השדות")
    fields = pd.DataFrame({
        "שדה":        ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age", "gender", "test_normal"],
        "משמעות":     ["דופק", "טמפרטורת גוף", "רמת חלבון", "תאי דם לבנים", "המוגלובין", "גיל", "מגדר (0/1)", "תוצאת הבדיקה"],
        "סוג נתון":   ["רציף", "רציף", "רציף", "רציף", "רציף", "בדיד", "בדיד", "בדיד"],
    })
    st.dataframe(fields, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="tip-box">
💡 <strong>המטרה שלנו:</strong> לחזות את <code>test_normal</code> — האם הבדיקה תקינה או לא.
הגרפים שנציג יעזרו לנו להבין <em>אילו שדות מבדילים</em> בין המקרים.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 💻 הקוד ליצירת ה-Dataset")
    st.code("""import streamlit as st
import pandas as pd
import numpy as np

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
    return df

df = generate_blood_test_dataset()
st.dataframe(df.head())
""", language="python")

# ══════════════════════════════════════════════
# SECTION 2 – Discrete vs Continuous
# ══════════════════════════════════════════════
elif section == "2️⃣  נתון בדיד vs. רציף":
    st.header("🔢 נתון בדיד לעומת נתון רציף")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🟡 נתון בדיד (Discrete)
<span class="tag-discrete">Discrete</span>

נתון שיכול לקבל **מספר קטן של ערכים קבועים**.

**דוגמאות ב-Dataset שלנו:**
- `test_normal` → רק `0` או `1`
- `gender` → רק `0` או `1`
- `age` → מספרים שלמים (18, 19, 20…)

**מה שואלים?**
> "כמה מטופלים בכל קטגוריה?"

**גרפים מתאימים: עמודות (Bar Chart)**
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
### 🔵 נתון רציף (Continuous)
<span class="tag-continuous">Continuous</span>

נתון שיכול לקבל **כל ערך** בטווח מסוים.

**דוגמאות ב-Dataset שלנו:**
- `heart_rate` → 60.0, 72.5, 95.3…
- `body_temp` → 36.2, 37.1, 38.4…
- `hemoglobin` → 11.5, 14.2…

**מה שואלים?**
> "מה ההתפלגות? האם יש הבדל בין קבוצות?"

**גרפים מתאימים: היסטוגרמה, פיזור, קופסה**
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 השוואה מהירה")

    compare = pd.DataFrame({
        "תכונה":        ["ערכים אפשריים", "דוגמה בנתונים", "גרף מומלץ", "שאלה טיפוסית"],
        "בדיד 🟡":      ["מספר קטן וקבוע", "gender: 0 או 1", "Bar Chart", "כמה מכל סוג?"],
        "רציף 🔵":      ["אינסוף ערכים", "heart_rate: 72.3", "Histogram / Scatter", "מה הטווח? יש הבדל?"],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="tip-box">
💡 <strong>כלל אצבע:</strong> אם אפשר לספור את מספר הערכים השונים על אצבעות — זה כנראה בדיד.
אם יש הרבה ערכים שונים (כמו דופק) — זה רציף.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 3 – Bar Chart
# ══════════════════════════════════════════════
elif section == "3️⃣  Bar Chart — עמודות":
    st.header("📊 Bar Chart — גרף עמודות")

    st.markdown("""
<div class="section-box">
<span class="tag-discrete">מתאים לנתון בדיד</span><br><br>
גרף עמודות מציג <strong>כמות</strong> של כל קטגוריה. הוא מושלם כאשר רוצים לראות
<em>כמה מטופלים תקינים לעומת לא תקינים</em>.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📌 דוגמה 1 — כמה מטופלים בכל תוצאה?")

    counts = df["label"].value_counts().reset_index()
    counts.columns = ["תוצאה", "כמות"]
    st.bar_chart(counts.set_index("תוצאה"))

    st.code("""# כמה מטופלים בכל קטגוריה?
counts = df["label"].value_counts().reset_index()
counts.columns = ["תוצאה", "כמות"]

st.bar_chart(counts.set_index("תוצאה"))
""", language="python")

    st.markdown("---")
    st.markdown("### 📌 דוגמה 2 — פילוח לפי מגדר")

    gender_counts = df.groupby(["gender", "label"]).size().reset_index(name="כמות")
    gender_pivot  = gender_counts.pivot(index="gender", columns="label", values="כמות").fillna(0)
    gender_pivot.index = ["נקבה (0)", "זכר (1)"]
    st.bar_chart(gender_pivot)

    st.code("""# כמה מכל מגדר — ולפי תוצאה?
gender_counts = df.groupby(["gender", "label"]).size().reset_index(name="כמות")
gender_pivot  = gender_counts.pivot(index="gender", columns="label", values="כמות").fillna(0)
gender_pivot.index = ["נקבה (0)", "זכר (1)"]

st.bar_chart(gender_pivot)
""", language="python")

    st.markdown("""
<div class="tip-box">
💡 <strong>מה למדנו?</strong> ניתן לראות אם יש הבדל בין מגדרים בתוצאות הבדיקה.
Bar Chart עוזר להשוות <em>כמויות</em> בין קטגוריות.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 4 – Line Chart
# ══════════════════════════════════════════════
elif section == "4️⃣  Line Chart — קווים":
    st.header("📈 Line Chart — גרף קווים")

    st.markdown("""
<div class="section-box">
<span class="tag-continuous">מתאים לנתון רציף</span><br><br>
גרף קווים מציג <strong>שינוי לאורך ציר X</strong> — בדרך כלל זמן, אבל יכול להיות גם מדד אחר.
כאן נשתמש בו כדי לראות את <em>הממוצע לפי גיל</em>.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📌 דוגמה — דופק ממוצע לפי גיל")

    feature = st.selectbox("בחר שדה להצגה:", ["heart_rate", "body_temp", "hemoglobin", "white_blood_cells"])

    age_avg = df.groupby("age")[feature].mean().reset_index()
    age_avg.columns = ["גיל", feature]
    st.line_chart(age_avg.set_index("גיל"))

    st.code(f"""# ממוצע {feature} לפי גיל
age_avg = df.groupby("age")["{feature}"].mean().reset_index()
age_avg.columns = ["גיל", "{feature}"]

st.line_chart(age_avg.set_index("גיל"))
""", language="python")

    st.markdown("""
<div class="tip-box">
💡 <strong>מתי משתמשים בקו?</strong> כאשר ציר ה-X הוא <em>סדרתי</em> — כלומר יש משמעות לסדר.
גיל, זמן, מדידה חוזרת. אם ציר X הוא קטגוריות ללא סדר — עדיף Bar Chart.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 5 – Scatter Chart
# ══════════════════════════════════════════════
elif section == "5️⃣  Scatter Chart — פיזור":
    st.header("🔵 Scatter Chart — גרף פיזור")

    st.markdown("""
<div class="section-box">
<span class="tag-continuous">מתאים לנתון רציף</span><br><br>
גרף פיזור מציג את <strong>הקשר בין שני משתנים</strong>. כל נקודה היא מטופל אחד.
נשתמש בו כדי לראות אם שני שדות ביחד מפרידים בין תקין ולא תקין.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("ציר X:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"], index=0)
    with col2:
        y_col = st.selectbox("ציר Y:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"], index=4)

    st.markdown("### 📌 נתקין לעומת לא תקין")

    normal_df   = df[df["test_normal"] == 1]
    abnormal_df = df[df["test_normal"] == 0]

    scatter_data = pd.DataFrame({
        f"תקין — {x_col}":    normal_df[x_col].values,
        f"תקין — {y_col}":    normal_df[y_col].values,
    }).dropna()

    # Streamlit scatter — מציג שני ערכים על אותו ציר X
    # נבנה DataFrame ידידותי
    plot_df = pd.DataFrame({
        "x":     df[x_col],
        "תקין":  df[y_col].where(df["test_normal"] == 1),
        "לא תקין": df[y_col].where(df["test_normal"] == 0),
    }).set_index("x").sort_index()

    st.scatter_chart(plot_df)

    st.code(f"""# גרף פיזור — {x_col} מול {y_col}, לפי תוצאה
plot_df = pd.DataFrame({{
    "x":          df["{x_col}"],
    "תקין":       df["{y_col}"].where(df["test_normal"] == 1),
    "לא תקין":   df["{y_col}"].where(df["test_normal"] == 0),
}}).set_index("x").sort_index()

st.scatter_chart(plot_df)
""", language="python")

    st.markdown("""
<div class="tip-box">
💡 <strong>מה אנחנו מחפשים?</strong> אם הנקודות הכחולות (תקין) והאדומות (לא תקין)
נמצאות באזורים שונים — זה אומר שהשדות האלו הם <em>מנבאים טובים</em> של test_normal!
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 6 – Histogram
# ══════════════════════════════════════════════
elif section == "6️⃣  Histogram — התפלגות":
    st.header("📉 Histogram — גרף התפלגות")

    st.markdown("""
<div class="section-box">
<span class="tag-continuous">מתאים לנתון רציף</span><br><br>
היסטוגרמה מחלקת את הערכים ל<strong>תאים (bins)</strong> ומציגה כמה ערכים נפלו בכל תא.
היא עוזרת להבין את <em>הצורה</em> של ההתפלגות — האם היא פעמון? מוטה? דו-פסגית?
</div>
""", unsafe_allow_html=True)

    feature = st.selectbox("בחר שדה:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])
    bins    = st.slider("כמות תאים (bins):", 5, 30, 15)

    st.markdown(f"### 📌 התפלגות של {feature} — תקין לעומת לא תקין")

    normal_vals   = df[df["test_normal"] == 1][feature]
    abnormal_vals = df[df["test_normal"] == 0][feature]

    # בניית histogram ידנית עם np.histogram ואז bar_chart
    all_min = df[feature].min()
    all_max = df[feature].max()
    edges   = np.linspace(all_min, all_max, bins + 1)

    n_counts, _ = np.histogram(normal_vals,   bins=edges)
    a_counts, _ = np.histogram(abnormal_vals, bins=edges)
    labels       = [f"{e:.1f}" for e in edges[:-1]]

    hist_df = pd.DataFrame({
        "תקין":    n_counts,
        "לא תקין": a_counts,
    }, index=labels)

    st.bar_chart(hist_df)

    st.code(f"""# היסטוגרמה — השוואת התפלגות בין תקין ולא תקין
import numpy as np

feature = "{feature}"
bins    = {bins}

normal_vals   = df[df["test_normal"] == 1][feature]
abnormal_vals = df[df["test_normal"] == 0][feature]

edges   = np.linspace(df[feature].min(), df[feature].max(), bins + 1)
n_counts, _ = np.histogram(normal_vals,   bins=edges)
a_counts, _ = np.histogram(abnormal_vals, bins=edges)
labels       = [f"{{e:.1f}}" for e in edges[:-1]]

hist_df = pd.DataFrame({{
    "תקין":    n_counts,
    "לא תקין": a_counts,
}}, index=labels)

st.bar_chart(hist_df)
""", language="python")

    st.markdown("""
<div class="tip-box">
💡 <strong>מה לחפש?</strong> אם עמודות "תקין" ו"לא תקין" חופפות הרבה — הנתון לא כל כך מפריד.
אם הן נמצאות באזורים שונים — הנתון הוא <em>מנבא טוב</em>!
נסה להחליף בין שדות ולראות מי מפריד הכי טוב.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 7 – Box Plot
# ══════════════════════════════════════════════
elif section == "7️⃣  Box Plot — תיבה":
    st.header("📦 Box Plot — גרף תיבה")

    st.markdown("""
<div class="section-box">
<span class="tag-continuous">מתאים לנתון רציף</span><br><br>
גרף תיבה (Box Plot) מסכם בצורה קומפקטית את <strong>ההתפלגות</strong> של נתון:
מינימום, Q1, חציון, Q3, מקסימום — וחריגים.
נבנה אותו ב-Streamlit באמצעות טבלת סטטיסטיקות + bar_chart.
</div>
""", unsafe_allow_html=True)

    feature = st.selectbox("בחר שדה:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])

    st.markdown(f"### 📌 סיכום סטטיסטי — {feature} לפי תוצאה")

    stats = df.groupby("label")[feature].describe()[["min", "25%", "50%", "75%", "max"]].T
    stats.index.name = "סטטיסטיקה"
    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### גרף ממוצע + רבעונים")
    box_df = df.groupby("label")[feature].agg(
        Q1=lambda x: x.quantile(0.25),
        חציון="median",
        Q3=lambda x: x.quantile(0.75),
    )
    st.bar_chart(box_df)

    st.code(f"""# Box Plot פשוט — סיכום סטטיסטי לפי קבוצה
feature = "{feature}"

# טבלת סטטיסטיקות
stats = df.groupby("label")[feature].describe()[["min","25%","50%","75%","max"]].T
st.dataframe(stats.style.format("{{:.2f}}"))

# גרף רבעונים
box_df = df.groupby("label")[feature].agg(
    Q1=lambda x: x.quantile(0.25),
    חציון="median",
    Q3=lambda x: x.quantile(0.75),
)
st.bar_chart(box_df)
""", language="python")

    st.markdown("""
<div class="tip-box">
💡 <strong>מה אומר לנו Box Plot?</strong><br>
אם ה<em>חציון</em> (median) של "תקין" ו"לא תקין" שונה מאוד — זה שדה חזק לחיזוי.<br>
שדה טוב = הקופסאות לא חופפות. שדה חלש = הקופסאות מאוד חופפות.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 איזה שדה הכי מבדיל?")
    st.markdown("השווה את ה-Q1 וה-Q3 בין הקבוצות:")

    summary_rows = []
    for col in ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin"]:
        med_norm   = df[df["test_normal"] == 1][col].median()
        med_abnorm = df[df["test_normal"] == 0][col].median()
        diff       = abs(med_norm - med_abnorm)
        summary_rows.append({"שדה": col, "חציון תקין": round(med_norm,1), "חציון לא תקין": round(med_abnorm,1), "הפרש": round(diff,1)})

    summary_df = pd.DataFrame(summary_rows).sort_values("הפרש", ascending=False)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.success(f"🥇 השדה המבדיל ביותר: **{summary_df.iloc[0]['שדה']}** עם הפרש של {summary_df.iloc[0]['הפרש']}")
