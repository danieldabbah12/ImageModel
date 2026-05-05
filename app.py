"""
🏠 אפליקציה חינוכית: חיזוי מחירי דירות
=========================================
אפליקציית Streamlit פשוטה לתלמידים שלומדים למידת מכונה בפעם הראשונה.
היא מלמדת איך בונים מודל שמנבא מחירי דירות, צעד אחר צעד.

הרצה:
    pip install streamlit pandas numpy matplotlib scikit-learn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# הגדרות כלליות + תמיכה ב-RTL (עברית)
# ============================================================
st.set_page_config(
    page_title="חיזוי מחירי דירות",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .main, .block-container, [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    h1, h2, h3, h4, h5, p, li, label, div { text-align: right; }
    .stRadio > div { direction: rtl; }
    .big-number {
        font-size: 38px;
        font-weight: bold;
        color: #1f77b4;
    }
    .step-box {
        background-color: #f0f8ff;
        padding: 16px;
        border-right: 5px solid #1f77b4;
        border-radius: 6px;
        margin: 10px 0;
    }
    .tip-box {
        background-color: #fff8dc;
        padding: 14px;
        border-right: 5px solid #ffa500;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# יצירת מאגר נתונים פשוט (סינתטי) - מחירי דירות
# ============================================================
@st.cache_data
def create_data(n: int = 200) -> pd.DataFrame:
    """יוצרים מאגר דירות מציאותי כדי שהתלמידים יוכלו להבין את הקשרים."""
    rng = np.random.default_rng(seed=42)
    size = rng.integers(40, 200, n)                       # גודל במ"ר
    rooms = np.clip(np.round(size / 25 + rng.normal(0, 0.5, n)), 1, 6).astype(int)
    age = rng.integers(0, 50, n)                          # גיל הבניין
    distance = np.round(rng.uniform(1, 30, n), 1)         # מרחק ממרכז העיר

    # נוסחה "אמיתית" שהמודל צריך ללמוד (בתוספת רעש)
    price = (
        size * 25_000
        + rooms * 50_000
        - age * 5_000
        - distance * 10_000
        + rng.normal(0, 100_000, n)
    )
    price = np.clip(price, 500_000, None).round(-3).astype(int)

    df = pd.DataFrame({
        "size_sqm": size,
        "rooms": rooms,
        "age_years": age,
        "distance_km": distance,
        "price_ils": price,
    })
    return df


HEBREW_NAMES = {
    "size_sqm": 'גודל (מ"ר)',
    "rooms": "מספר חדרים",
    "age_years": "גיל הבניין (שנים)",
    "distance_km": 'מרחק ממרכז העיר (ק"מ)',
    "price_ils": 'מחיר (ש"ח)',
}

df = create_data()
features = ["size_sqm", "rooms", "age_years", "distance_km"]
target = "price_ils"


# ============================================================
# אימון המודל פעם אחת ושמירה במטמון
# ============================================================
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred


model, X_train, X_test, y_train, y_test, y_pred = train_model(df)


# ============================================================
# סרגל הניווט
# ============================================================
st.sidebar.title("🏠 סרגל ניווט")
st.sidebar.markdown("בחרו שלב כדי ללמוד יחד צעד אחר צעד 👇")

page = st.sidebar.radio(
    "שלבי הלמידה:",
    [
        "1️⃣ מבוא — מה זה Machine Learning?",
        "2️⃣ הכרת הנתונים",
        "3️⃣ חקירה ויזואלית",
        "4️⃣ הכנת הנתונים למודל",
        "5️⃣ אימון המודל",
        "6️⃣ הערכת המודל",
        "7️⃣ נסו בעצמכם — חיזוי בלייב!",
        "8️⃣ סיכום ומה הלאה?",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **טיפ:** התקדמו בסדר השלבים — כל שלב נבנה על הקודם.\n\n"
    "הקובץ הוא `app.py` יחיד, ניתן לפתוח ולקרוא את הקוד."
)


# ============================================================
# 1️⃣ מבוא
# ============================================================
if page.startswith("1"):
    st.title("🏠 חיזוי מחירי דירות בעזרת מחשב")
    st.markdown("### ברוכים הבאים לעולם ה-Machine Learning! 🤖")

    st.markdown(
        """
        <div class="step-box">
        <b>השאלה שלנו היום:</b><br>
        אם אני מספר לכם שיש דירה בגודל 90 מ"ר, עם 3 חדרים, בת 10 שנים,
        במרחק 5 ק"מ ממרכז העיר — <b>כמה היא צריכה לעלות?</b> 🤔
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("מה זה בעצם Machine Learning?")
    st.markdown(
        """
        דמיינו שאתם רואים הרבה דירות עם המחירים שלהן.
        אחרי כמה דוגמאות אתם מתחילים להרגיש "מה הגיוני":

        - דירה גדולה יותר ➜ עולה יותר 💰
        - דירה ישנה יותר ➜ עולה פחות 📉
        - דירה רחוקה ממרכז העיר ➜ עולה פחות 🛣️

        **למידת מכונה (Machine Learning)** היא בדיוק זה —
        אנחנו נותנים למחשב הרבה דוגמאות, והוא לומד את הכללים בעצמו.
        """
    )

    st.subheader("מה נעשה בפרויקט הזה?")
    cols = st.columns(4)
    cols[0].markdown("#### 1. נתונים 📊")
    cols[0].write("נסתכל על דירות שכבר נמכרו.")
    cols[1].markdown("#### 2. חקירה 🔍")
    cols[1].write("נבין מי משפיע על המחיר.")
    cols[2].markdown("#### 3. אימון 🧠")
    cols[2].write("נלמד את המחשב לחזות.")
    cols[3].markdown("#### 4. חיזוי 🎯")
    cols[3].write("נבדוק כמה הוא טוב.")

    st.markdown(
        """
        <div class="tip-box">
        🎯 בסוף האפליקציה תוכלו להזין נתונים של דירה משלכם
        והמחשב יגיד לכם <b>כמה הוא חושב שהיא שווה</b>!
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 2️⃣ הכרת הנתונים
# ============================================================
elif page.startswith("2"):
    st.title("2️⃣ הכרת הנתונים 📊")
    st.markdown(
        """
        כל פרויקט של Machine Learning מתחיל ב**נתונים**.
        לפנינו מאגר של 200 דירות שנמכרו, יחד עם מאפיינים שונים שלהן.
        """
    )

    st.subheader("מה יש בטבלה?")
    st.markdown(
        """
        - **גודל (מ"ר)** — כמה גדולה הדירה.
        - **מספר חדרים** — כמה חדרים יש בה.
        - **גיל הבניין (שנים)** — כמה שנים הבניין קיים.
        - **מרחק ממרכז העיר (ק"מ)** — כמה רחוקה הדירה מהמרכז.
        - **מחיר (ש"ח)** — *זה מה שאנחנו רוצים לחזות!* 🎯
        """
    )

    st.subheader("כך נראים הנתונים שלנו (10 דירות ראשונות):")
    st.dataframe(
        df.head(10).rename(columns=HEBREW_NAMES),
        use_container_width=True,
    )

    st.subheader("📈 סטטיסטיקה בסיסית")
    st.markdown("בואו נראה מה הממוצע, המינימום, המקסימום של כל עמודה:")
    st.dataframe(
        df.describe().round(0).rename(columns=HEBREW_NAMES),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("מספר דירות", f"{len(df):,}")
    c2.metric('מחיר ממוצע (ש"ח)', f"{int(df[target].mean()):,}")
    c3.metric('גודל ממוצע (מ"ר)', f"{int(df['size_sqm'].mean())}")

    st.markdown(
        """
        <div class="tip-box">
        🧠 <b>מושג חשוב:</b><br>
        - <b>Features (תכונות / X):</b> הדברים שאנחנו <i>יודעים</i> על הדירה
        (גודל, חדרים, גיל, מרחק).<br>
        - <b>Target (מטרה / y):</b> הדבר שאנחנו <i>רוצים לחזות</i> — המחיר.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3️⃣ חקירה ויזואלית
# ============================================================
elif page.startswith("3"):
    st.title("3️⃣ חקירה ויזואלית 🔍")
    st.markdown(
        """
        לפני שאנחנו בונים מודל, חשוב לראות **תמונות** של הנתונים.
        ככה נבין אילו תכונות באמת משפיעות על המחיר.
        """
    )

    st.subheader("איך כל תכונה קשורה למחיר?")
    feature_he = st.selectbox(
        "בחרו תכונה כדי לראות את הקשר שלה למחיר:",
        options=features,
        format_func=lambda x: HEBREW_NAMES[x],
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df[feature_he], df[target], alpha=0.6, color="#1f77b4")
    ax.set_xlabel(HEBREW_NAMES[feature_he][::-1])  # היפוך לעברית במטפלוטליב
    ax.set_ylabel(HEBREW_NAMES[target][::-1])
    ax.set_title(("קשר בין " + HEBREW_NAMES[feature_he] + " למחיר")[::-1])
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown(
        """
        <div class="step-box">
        🟢 אם הנקודות עולות משמאל לימין ➜ ככל שהתכונה גדלה, המחיר עולה (קשר חיובי).<br>
        🔴 אם הנקודות יורדות ➜ ככל שהתכונה גדלה, המחיר יורד (קשר שלילי).<br>
        ⚪ אם הנקודות מפוזרות אקראית ➜ אין קשר משמעותי.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 מטריצת מתאמים (Correlation)")
    st.markdown(
        "מספר בין -1 ל-1 שמראה כמה שתי תכונות קשורות. "
        "ככל שהוא קרוב ל-1 או -1 — הקשר חזק יותר."
    )
    corr = df.corr().round(2)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    cax = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig2.colorbar(cax)
    st.pyplot(fig2)

    st.markdown(
        """
        <div class="tip-box">
        🔎 <b>מה לחפש?</b> בעמודת <code>price_ils</code> — איזו תכונה הכי
        מתואמת עם המחיר? (רמז: תסתכלו מי הכי קרוב ל-1 או ל--1)
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4️⃣ הכנת הנתונים למודל
# ============================================================
elif page.startswith("4"):
    st.title("4️⃣ הכנת הנתונים למודל 🧹")

    st.markdown(
        """
        לפני שמלמדים את המחשב, צריך לארגן את הנתונים בצורה מסודרת.
        אנחנו עושים שני דברים:
        """
    )

    st.subheader("שלב א: מפרידים בין X (תכונות) ל-y (מטרה)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**X — מה שיודעים על הדירה:**")
        st.dataframe(df[features].head().rename(columns=HEBREW_NAMES), use_container_width=True)
    with c2:
        st.markdown("**y — מה שרוצים לחזות (המחיר):**")
        st.dataframe(df[[target]].head().rename(columns=HEBREW_NAMES), use_container_width=True)

    st.subheader("שלב ב: מחלקים לאימון ומבחן (Train / Test)")
    st.markdown(
        """
        זה אחד **הרעיונות הכי חשובים** בלמידת מכונה!

        - 📚 **Train (אימון - 80%)** — דוגמאות שהמודל רואה ולומד מהן.
        - 📝 **Test (מבחן - 20%)** — דוגמאות שהמודל **לא ראה**, ובהן בודקים אם הוא באמת למד.

        זה בדיוק כמו בית ספר: לומדים על דוגמאות במחברת,
        ואז עושים מבחן על שאלות חדשות. אם תלמיד יראה את המבחן מראש — אי אפשר לדעת אם הוא באמת מבין!
        """
    )

    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.barh([0], [80], color="#2ca02c", label="Train (80%)")
    ax.barh([0], [20], left=[80], color="#d62728", label="Test (20%)")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("אחוז מהנתונים"[::-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.metric("גודל קבוצת אימון", f"{len(X_train)} דירות")
    c2.metric("גודל קבוצת מבחן", f"{len(X_test)} דירות")

    with st.expander("📜 הקוד שעשה את זה"):
        st.code(
            """
from sklearn.model_selection import train_test_split

X = df[["size_sqm", "rooms", "age_years", "distance_km"]]
y = df["price_ils"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
            """,
            language="python",
        )


# ============================================================
# 5️⃣ אימון המודל
# ============================================================
elif page.startswith("5"):
    st.title("5️⃣ אימון המודל 🧠")

    st.markdown(
        """
        עכשיו לחלק הכייפי! נלמד את המחשב לחזות מחירים.
        נשתמש במודל הכי פשוט וידידותי שיש: **רגרסיה לינארית (Linear Regression)**.
        """
    )

    st.subheader("מה זה רגרסיה לינארית? 📐")
    st.markdown(
        r"""
        המודל מנסה למצוא נוסחה פשוטה בצורה הזאת:

        $$
        \text{מחיר} = w_1 \cdot \text{גודל} + w_2 \cdot \text{חדרים}
                  + w_3 \cdot \text{גיל} + w_4 \cdot \text{מרחק} + b
        $$

        - כל $w$ הוא **משקל** — כמה התכונה משפיעה על המחיר.
        - $b$ זה **הקבוע** (intercept) — נקודת ההתחלה.

        המודל מנסה הרבה ערכים שונים של $w$ ו-$b$, עד שהוא מוצא את הצירוף
        שמסביר הכי טוב את הנתונים שלנו. 🪄
        """
    )

    st.subheader("המשקלים שהמודל למד מהנתונים שלנו:")
    weights = pd.DataFrame(
        {
            "תכונה": [HEBREW_NAMES[f] for f in features],
            "משקל (w)": np.round(model.coef_, 1),
        }
    )
    st.dataframe(weights, use_container_width=True)

    st.write(f"**הקבוע (b):** `{model.intercept_:,.0f}` ש\"ח")

    st.markdown(
        """
        <div class="tip-box">
        🧠 <b>איך קוראים את הטבלה?</b><br>
        משקל חיובי גדול ➜ התכונה <b>מעלה</b> את המחיר.<br>
        משקל שלילי ➜ התכונה <b>מורידה</b> את המחיר.<br>
        לדוגמה: כל מ"ר נוסף מוסיף לדירה בערך כך וכך שקלים. 💸
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📜 הקוד שאימן את המודל"):
        st.code(
            """
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)   # ← זה השלב שבו המחשב לומד!
            """,
            language="python",
        )


# ============================================================
# 6️⃣ הערכת המודל
# ============================================================
elif page.startswith("6"):
    st.title("6️⃣ הערכת המודל 🎯")
    st.markdown(
        "אימנו את המודל על קבוצת ה-Train. עכשיו נבדוק עליו את "
        "**קבוצת המבחן (Test)** — דירות שהוא מעולם לא ראה."
    )

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    c1.markdown("#### MAE — שגיאה ממוצעת")
    c1.markdown(f'<div class="big-number">{mae:,.0f} ש"ח</div>', unsafe_allow_html=True)
    c1.caption("בממוצע, כמה המודל מפספס במחיר של דירה.")

    c2.markdown("#### R² — איכות המודל")
    c2.markdown(f'<div class="big-number">{r2:.2%}</div>', unsafe_allow_html=True)
    c2.caption("ככל שהמספר קרוב ל-100%, המודל מסביר טוב יותר את המחיר.")

    st.subheader("🔭 חיזוי מול מציאות")
    st.markdown(
        "כל נקודה היא דירה אחת מקבוצת המבחן. אם המודל היה מושלם, "
        "כל הנקודות היו על הקו האדום."
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4", label="חיזויים")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="מודל מושלם")
    ax.set_xlabel("מחיר אמיתי"[::-1])
    ax.set_ylabel("מחיר שהמודל ניבא"[::-1])
    ax.set_title("חיזוי מול מציאות"[::-1])
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("📋 דוגמאות מתוך המבחן")
    sample = pd.DataFrame({
        'מחיר אמיתי (ש"ח)': y_test.values[:10].astype(int),
        'חיזוי המודל (ש"ח)': y_pred[:10].astype(int),
        'הפרש (ש"ח)': (y_pred[:10] - y_test.values[:10]).astype(int),
    })
    st.dataframe(sample, use_container_width=True)

    st.markdown(
        """
        <div class="tip-box">
        🤔 <b>שאלה לתלמידים:</b> איך אפשר לשפר את המודל?<br>
        - להוסיף עוד תכונות (למשל: יש מעלית? קומה?)<br>
        - לאסוף יותר דירות לאימון.<br>
        - לנסות מודלים מתקדמים יותר (Decision Tree, Random Forest…).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7️⃣ נסו בעצמכם
# ============================================================
elif page.startswith("7"):
    st.title("7️⃣ נסו בעצמכם — חיזוי בלייב! 🪄")
    st.markdown(
        "הזינו פרטים של דירה דמיונית, והמודל יגיד לכם כמה לדעתו היא שווה."
    )

    c1, c2 = st.columns(2)
    with c1:
        size = st.slider('גודל הדירה (מ"ר)', 30, 220, 90, step=5)
        rooms = st.slider("מספר חדרים", 1, 7, 3)
    with c2:
        age = st.slider("גיל הבניין (שנים)", 0, 60, 10)
        distance = st.slider('מרחק ממרכז העיר (ק"מ)', 0.0, 35.0, 5.0, step=0.5)

    new_apartment = pd.DataFrame(
        [[size, rooms, age, distance]],
        columns=features,
    )
    prediction = float(model.predict(new_apartment)[0])

    st.markdown("### 🎯 לפי המודל, הדירה שלכם שווה:")
    st.markdown(
        f'<div class="big-number">{prediction:,.0f} ש"ח</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🧮 איך הגענו למספר הזה?")
    breakdown = pd.DataFrame(
        {
            "תכונה": [HEBREW_NAMES[f] for f in features],
            "ערך שהזנתם": [size, rooms, age, distance],
            "משקל (w)": np.round(model.coef_, 1),
            'תרומה למחיר (ש"ח)': np.round(
                model.coef_ * np.array([size, rooms, age, distance]), 0
            ).astype(int),
        }
    )
    st.dataframe(breakdown, use_container_width=True)
    st.write(f'➕ קבוע (b): **{model.intercept_:,.0f} ש"ח**')
    st.write(f'🟰 סכום הכול: **{prediction:,.0f} ש"ח**')

    st.markdown(
        """
        <div class="tip-box">
        🎲 <b>נסו לשחק:</b><br>
        - מה קורה כשמגדילים את הדירה ב-50 מ"ר?<br>
        - מה הדירה היקרה ביותר שאפשר לבנות עם הסליידרים?<br>
        - מה הזולה ביותר?
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8️⃣ סיכום
# ============================================================
elif page.startswith("8"):
    st.title("8️⃣ סיכום ומה הלאה? 🚀")

    st.markdown(
        """
        ### מה למדנו היום? 🎓
        1. ❓ הגדרנו **בעיה** — לחזות מחירי דירות.
        2. 📊 בדקנו את **הנתונים** והבנו מה יש בהם.
        3. 🔍 **חקרנו** ויזואלית את הקשר בין תכונות למחיר.
        4. 🧹 **הכנו** את הנתונים — חילקנו ל-X / y ול-Train / Test.
        5. 🧠 **אימנו** מודל רגרסיה לינארית.
        6. 🎯 **הערכנו** את המודל בעזרת MAE ו-R².
        7. 🪄 השתמשנו במודל כדי **לחזות** מחיר של דירה חדשה.
        """
    )

    st.markdown(
        """
        ### צעדים הבאים מומלצים 🌱
        - להחליף את הדאטה במאגר אמיתי (למשל, House Prices מ-Kaggle).
        - לנסות מודלים אחרים: `DecisionTreeRegressor`, `RandomForestRegressor`.
        - להוסיף עוד תכונות: קומה, חניה, מעלית, אזור עירוני וכו'.
        - לבדוק את ההבדל בין ביצועי Train ל-Test (Overfitting!).
        """
    )

    st.markdown(
        """
        <div class="step-box">
        💬 <b>הרעיון הכי חשוב לזכור:</b><br>
        Machine Learning זה לא קסם — זה <b>למצוא דפוסים בנתונים</b>.<br>
        ככל שהנתונים שלנו טובים ומגוונים יותר, המודל ילמד טוב יותר.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.balloons()
