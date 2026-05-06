import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="מדריך ויזואליזציה",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Rubik', sans-serif;
    direction: rtl;
}
h1, h2, h3 { font-family: 'Rubik', sans-serif; font-weight: 700; }
code, pre {
    font-family: 'Fira Code', monospace !important;
    direction: ltr;
    text-align: left;
}
.section-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin: 1.5rem 0 1rem 0;
    font-size: 1.3rem;
    font-weight: 700;
    border-left: 5px solid #e94560;
}
.tip-box {
    background: #e8f4fd;
    border-right: 4px solid #2196F3;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    color: #1a1a2e;
}
.concept-box {
    background: #fff8e1;
    border-right: 4px solid #FFC107;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    color: #1a1a2e;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)


# ─── Generate Synthetic DataFrame ─────────────────────────────────────────────
@st.cache_data
def generate_df():
    np.random.seed(42)
    n = 200
    target = np.random.choice([0, 1], size=n, p=[0.55, 0.45])
    df = pd.DataFrame({
        "גיל":          np.where(target == 1,
                            np.random.normal(45, 12, n),
                            np.random.normal(35, 10, n)).astype(int).clip(18, 80),
        "הכנסה":        np.where(target == 1,
                            np.random.normal(8000, 2000, n),
                            np.random.normal(6000, 1500, n)).clip(2000, 20000).astype(int),
        "שעות_שינה":    np.random.normal(7, 1.2, n).clip(3, 10).round(1),
        "פעילות_גופנית": np.random.normal(3, 1.5, n).clip(0, 7).round(1),
        "לחץ_דם":       np.where(target == 1,
                            np.random.normal(135, 15, n),
                            np.random.normal(120, 10, n)).clip(90, 180).astype(int),
        "BMI":          np.where(target == 1,
                            np.random.normal(27, 4, n),
                            np.random.normal(23, 3, n)).clip(16, 45).round(1),
        "רמת_סוכר":     np.where(target == 1,
                            np.random.normal(115, 25, n),
                            np.random.normal(90, 15, n)).clip(60, 200).astype(int),
        "עישון":        np.random.choice(["לא מעשן", "לשעבר", "מעשן"], size=n, p=[0.5, 0.3, 0.2]),
        "מין":          np.random.choice(["זכר", "נקבה"], size=n),
        "מחלה":         np.where(target == 1, "חולה", "בריא"),   # ← משתנה מטרה
    })
    return df

df = generate_df()
TARGET_COL = "מחלה"
NUMERIC_COLS = [c for c in df.select_dtypes(include=np.number).columns]
CAT_COLS = [c for c in df.select_dtypes(include="object").columns if c != TARGET_COL]


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 מדריך ויזואליזציה")
    st.markdown("---")
    section = st.radio(
        "בחר נושא:",
        [
            "🏠 מבוא וה-DataFrame",
            "🔥 Heatmap",
            "📊 Count Plot",
            "🔵 Scatter Plot",
            "🔮 לקראת KNN",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**משתנה מטרה:** `מחלה` (חולה / בריא)")
    st.markdown("**גודל ה-DataFrame:** 200 שורות × 10 עמודות")


def show_code(code: str):
    st.code(code, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# 🏠 מבוא
# ══════════════════════════════════════════════════════════════════════════════
if section == "🏠 מבוא וה-DataFrame":
    st.title("📊 ויזואליזציה של נתונים לקראת KNN")
    st.subheader("איך לחקור DataFrame עם משתנה מטרה בעזרת גרפים")

    st.markdown("""
    במדריך זה נעבוד עם **DataFrame סינתטי** בנושא בריאות.  
    המטרה: להבין את הנתונים לפני שמריצים אלגוריתם **KNN**.
    
    **3 כלי הגרפיקה שנלמד:**
    | גרף | שאלה שהוא עונה |
    |-----|---------------|
    | 🔥 Heatmap | איזה משתנים קשורים זה לזה? |
    | 📊 Count Plot | איך מתחלק משתנה קטגוריאלי? |
    | 🔵 Scatter Plot | מה הקשר בין 2 משתנים + המטרה? |
    """)

    st.markdown("---")
    st.markdown("### ה-DataFrame שנשתמש בו")

    show_code("""import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

# יצירת נתונים סינתטיים
target = np.random.choice([0, 1], size=n, p=[0.55, 0.45])

df = pd.DataFrame({
    "גיל":           ...,   # מספרי
    "הכנסה":         ...,   # מספרי
    "שעות_שינה":     ...,   # מספרי
    "פעילות_גופנית": ...,   # מספרי
    "לחץ_דם":        ...,   # מספרי
    "BMI":           ...,   # מספרי
    "רמת_סוכר":      ...,   # מספרי
    "עישון":         ...,   # קטגוריאלי
    "מין":           ...,   # קטגוריאלי
    "מחלה":          ...,   # משתנה מטרה ✅
})""")

    st.markdown("#### תצוגה מקדימה:")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("סה\"כ שורות", len(df))
    col2.metric("חולים", (df[TARGET_COL] == "חולה").sum())
    col3.metric("בריאים", (df[TARGET_COL] == "בריא").sum())

    st.markdown("---")
    st.markdown("#### סטטיסטיקה בסיסית:")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔥 Heatmap
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔥 Heatmap":
    st.markdown('<div class="section-header">🔥 Heatmap — מטריצת קורלציות</div>', unsafe_allow_html=True)

    st.markdown("""
    ### מה זה Heatmap?
    **Heatmap של קורלציות** מציג את **עוצמת הקשר** בין כל זוג משתנים מספריים.
    
    - ערך קרוב ל-**1**: קורלציה חיובית חזקה (כשאחד עולה, השני גם)  
    - ערך קרוב ל-**-1**: קורלציה שלילית חזקה (כשאחד עולה, השני יורד)  
    - ערך קרוב ל-**0**: כמעט אין קשר
    """)

    st.markdown('<div class="concept-box">💡 <b>למה חשוב לפני KNN?</b> KNN מבוסס על מרחק בין נקודות. אם שני משתנים מאוד מקורלציים, הם "אומרים אותו דבר" וייתכן שנרצה להסיר אחד מהם.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### הקוד")

    show_code("""import matplotlib.pyplot as plt
import seaborn as sns

# 1. בחר רק עמודות מספריות
numeric_df = df.select_dtypes(include='number')

# 2. חשב קורלציות
corr_matrix = numeric_df.corr()

# 3. צור את הגרף
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,         # הצג את הערכים בתוך כל תא
    fmt=".2f",          # שתי ספרות אחרי הנקודה
    cmap="coolwarm",    # צבעים: כחול=שלילי, אדום=חיובי
    center=0,           # 0 יהיה לבן/ניטרלי
    square=True,        # תאים מרובעים
    linewidths=0.5,
    ax=ax
)
ax.set_title("מטריצת קורלציות", fontsize=16, pad=15)
plt.tight_layout()

# 4. הצג ב-Streamlit
st.pyplot(fig)""")

    st.markdown("### פלט:")

    numeric_df = df[NUMERIC_COLS]
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("מטריצת קורלציות", fontsize=16, pad=15)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### פרשנות הגרף")
    
    # Find top correlations
    corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_pairs.append((cols[i], cols[j], abs(corr_matrix.iloc[i, j])))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    st.markdown("**הקורלציות החזקות ביותר:**")
    top_corr_data = [{"משתנה 1": p[0], "משתנה 2": p[1], "קורלציה": f"{corr_matrix.loc[p[0], p[1]]:.2f}"} for p in corr_pairs[:5]]
    st.table(pd.DataFrame(top_corr_data))

    st.markdown('<div class="tip-box">💡 <b>טיפ:</b> הקורלציה על האלכסון תמיד 1.0 — כל משתנה מקורלצי עם עצמו.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 📊 Count Plot
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Count Plot":
    st.markdown('<div class="section-header">📊 Count Plot — ספירת קטגוריות</div>', unsafe_allow_html=True)

    st.markdown("""
    ### מה זה Count Plot?
    **Count Plot** סופר את מספר ההופעות של כל ערך בעמודה קטגוריאלית,
    ומאפשר לפצל (hue) לפי **משתנה המטרה**.
    
    - ציר X: הקטגוריות  
    - ציר Y: כמות השורות  
    - **צבע (hue):** משתנה המטרה — חולה / בריא
    """)

    st.markdown('<div class="concept-box">💡 <b>למה חשוב לפני KNN?</b> KNN לא מטפל בעמודות קטגוריאליות ישירות. Count Plot עוזר להחליט אם לעשות One-Hot Encoding ועד כמה המשתנה חשוב.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Interactive: choose which categorical column
    chosen_cat = st.selectbox("בחר עמודה קטגוריאלית:", CAT_COLS)

    st.markdown("### הקוד")
    show_code(f"""import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=df,
    x="{chosen_cat}",        # העמודה שרוצים לספור
    hue="מחלה",              # פיצול לפי משתנה מטרה
    palette={{"בריא": "#4CAF50", "חולה": "#e94560"}},
    ax=ax
)

ax.set_title(f"התפלגות {chosen_cat} לפי מחלה", fontsize=14)
ax.set_xlabel("{chosen_cat}")
ax.set_ylabel("כמות")
ax.legend(title="מחלה")
plt.tight_layout()

st.pyplot(fig)""")

    st.markdown("### פלט:")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=chosen_cat,
        hue=TARGET_COL,
        palette={"בריא": "#4CAF50", "חולה": "#e94560"},
        ax=ax
    )
    ax.set_title(f"התפלגות {chosen_cat} לפי מחלה", fontsize=14)
    ax.set_xlabel(chosen_cat)
    ax.set_ylabel("כמות")
    ax.legend(title="מחלה")
    plt.tight_layout()
    st.pyplot(fig)

    # Also show st.bar_chart version
    st.markdown("---")
    st.markdown("### גרסה ב-Streamlit Native (`st.bar_chart`)")
    st.markdown("אפשר גם ישירות עם Streamlit — פחות שליטה על עיצוב אבל מהיר יותר:")
    show_code(f"""counts = df.groupby(["{chosen_cat}", "מחלה"]).size().unstack(fill_value=0)
st.bar_chart(counts)""")
    counts = df.groupby([chosen_cat, TARGET_COL]).size().unstack(fill_value=0)
    st.bar_chart(counts)

    st.markdown('<div class="tip-box">💡 <b>טיפ:</b> אם קטגוריה מסוימת מופיעה כמעט רק ב"חולה" — זה סימן שהיא מנבא חזק!</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔵 Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔵 Scatter Plot":
    st.markdown('<div class="section-header">🔵 Scatter Plot — פיזור עם צבע מטרה</div>', unsafe_allow_html=True)

    st.markdown("""
    ### מה זה Scatter Plot?
    **Scatter Plot** מציב כל שורה כנקודה במישור דו-ממדי:
    
    - ציר X: משתנה מספרי ראשון  
    - ציר Y: משתנה מספרי שני  
    - **צבע הנקודה:** משתנה המטרה — חולה / בריא
    
    כך ניתן לראות **אם ניתן להפריד** בין הקבוצות במישור הזה — בדיוק מה ש-KNN עושה!
    """)

    st.markdown('<div class="concept-box">💡 <b>למה חשוב לפני KNN?</b> אם הנקודות של "חולה" ו"בריא" מופרדות יפה — KNN יצליח שם. אם הן מעורבות — KNN יתקשה.</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("ציר X:", NUMERIC_COLS, index=0)
    with col2:
        y_col = st.selectbox("ציר Y:", NUMERIC_COLS, index=4)

    st.markdown("### הקוד")
    show_code(f"""import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="{x_col}",
    y="{y_col}",
    hue="מחלה",                # צבע = משתנה מטרה
    palette={{"בריא": "#4CAF50", "חולה": "#e94560"}},
    alpha=0.7,                  # שקיפות — עוזר לראות חפיפות
    s=60,                       # גודל נקודה
    ax=ax
)

ax.set_title(f"{x_col} vs {y_col} — לפי מחלה", fontsize=14)
ax.set_xlabel("{x_col}")
ax.set_ylabel("{y_col}")
ax.legend(title="מחלה")
plt.tight_layout()

st.pyplot(fig)""")

    st.markdown("### פלט:")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=TARGET_COL,
        palette={"בריא": "#4CAF50", "חולה": "#e94560"},
        alpha=0.7,
        s=60,
        ax=ax
    )
    ax.set_title(f"{x_col} vs {y_col} — לפי מחלה", fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title="מחלה")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Pair Plot — כל הזוגות בבת אחת")
    st.markdown("אם רוצים לראות את כל הגרפים בבת אחת:")
    show_code("""# בחר רק כמה עמודות (pair plot עם הרבה עמודות יהיה צפוף)
selected = ["גיל", "BMI", "לחץ_דם", "רמת_סוכר", "מחלה"]
subset = df[selected]

fig = sns.pairplot(subset, hue="מחלה",
                   palette={"בריא": "#4CAF50", "חולה": "#e94560"},
                   plot_kws={"alpha": 0.6})
st.pyplot(fig)""")

    if st.button("🔍 הצג Pair Plot (לוקח שנייה)"):
        with st.spinner("מחשב pair plot..."):
            selected_cols = ["גיל", "BMI", "לחץ_דם", "רמת_סוכר", TARGET_COL]
            subset = df[selected_cols]
            pair_fig = sns.pairplot(
                subset,
                hue=TARGET_COL,
                palette={"בריא": "#4CAF50", "חולה": "#e94560"},
                plot_kws={"alpha": 0.6}
            )
            st.pyplot(pair_fig)

    st.markdown('<div class="tip-box">💡 <b>טיפ:</b> שנה את ציר X ו-Y למעלה וחפש זוג שבו הנקודות הירוקות והאדומות מופרדות — שם KNN יצליח הכי טוב!</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🔮 לקראת KNN
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 לקראת KNN":
    st.markdown('<div class="section-header">🔮 לקראת KNN — מה למדנו מהגרפים?</div>', unsafe_allow_html=True)

    st.markdown("""
    ### סיכום התובנות מהויזואליזציה
    
    לפני שמריצים KNN, הגרפים שלמדנו עוזרים לנו לענות על 3 שאלות חיוניות:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔥 Heatmap אמר לנו:**
        - אילו משתנים מקורלציים?
        - האם כדאי להסיר עמודות כפולות?
        - מה ה"מידע הנוסף האמיתי" שכל עמודה מוסיפה?
        """)

    with col2:
        st.markdown("""
        **📊 Count Plot אמר לנו:**
        - האם המשתנים הקטגוריאליים מפרידים בין חולה/בריא?
        - אם כן — חשוב לעשות להם Encoding!
        - האם יש חוסר איזון בקבוצות?
        """)

    with col3:
        st.markdown("""
        **🔵 Scatter Plot אמר לנו:**
        - באיזה מרחב הנתונים ניתנים להפרדה?
        - KNN יעבוד טוב על אותם צירים
        - מה כדאי לנרמל לפני KNN?
        """)

    st.markdown("---")
    st.markdown("### צ'קליסט לפני KNN")

    checks = {
        "✅ ביצעתי Heatmap וזיהיתי קורלציות גבוהות": True,
        "✅ בדקתי Count Plot למשתנים קטגוריאליים": True,
        "✅ ביצעתי Scatter Plot לזוגות משתנים": True,
        "⬜ נרמלתי את הנתונים (StandardScaler / MinMaxScaler)": False,
        "⬜ עשיתי One-Hot Encoding למשתנים קטגוריאליים": False,
        "⬜ חילקתי ל-Train / Test": False,
        "⬜ בחרתי K מתאים": False,
    }

    for item, done in checks.items():
        if done:
            st.success(item)
        else:
            st.warning(item)

    st.markdown("---")
    st.markdown("### קוד בסיסי ל-KNN (Preview)")
    show_code("""from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. הכנת X ו-y
X = df[["גיל", "BMI", "לחץ_דם", "רמת_סוכר", "הכנסה"]]
y = (df["מחלה"] == "חולה").astype(int)

# 2. נרמול — חובה ב-KNN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. חלוקה ל-Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. אימון
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 5. הערכה
accuracy = knn.score(X_test, y_test)
st.metric("דיוק", f"{accuracy:.1%}")""")

    st.markdown('<div class="tip-box">💡 <b>חשוב:</b> KNN רגיש מאוד לסקאלה! תמיד יש לנרמל את הנתונים לפני האימון.</div>', unsafe_allow_html=True)
