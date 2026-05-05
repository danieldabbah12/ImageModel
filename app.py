import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

# ───────────────────────────────────────────────
st.set_page_config(page_title="סיווג תמונות – למתחילים", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap');
* { font-family: 'Heebo', sans-serif; direction: rtl; }
body, .stApp { background: #f7f5f0; color: #1a1a1a; }

h1 { font-size: 2.2rem; font-weight: 900; color: #1a1a1a; }
h2 { font-size: 1.4rem; font-weight: 700; color: #1a1a1a; border-bottom: 3px solid #1a1a1a; padding-bottom: 6px; }

.card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border: 1.5px solid #e0ddd8;
    box-shadow: 2px 3px 0px #d0ccc4;
}
.step-number {
    display: inline-block;
    width: 32px; height: 32px;
    background: #1a1a1a; color: white;
    border-radius: 50%;
    font-weight: 900; font-size: 1rem;
    text-align: center; line-height: 32px;
    margin-left: 8px;
}
.tag {
    display: inline-block;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    background: #1a1a1a; color: white;
    margin-bottom: 10px;
}
.result-box {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 900;
}
.wrong-box {
    background: #fff7ed;
    border: 2px solid #f97316;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 900;
}
.tip { background: #fffbeb; border-right: 4px solid #f59e0b; padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; font-size: 0.9rem; margin: 0.8rem 0; }
.code-area { background: #1a1a1a; color: #a8ff78; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.82rem; line-height: 1.7; white-space: pre; overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  פונקציות עזר – מוסתרות מהתלמיד, פשוטות מאוד
# ═══════════════════════════════════════════════════════════

def make_image(kind, seed=0):
    """מייצר תמונה פשוטה: ⭕ עיגול = חתול  | ▭ ריבוע = כלב"""
    random.seed(seed); np.random.seed(seed)
    img = Image.new("RGB", (80, 80), (230, 225, 215))
    d = ImageDraw.Draw(img)
    if kind == "cat":
        d.ellipse([15, 15, 65, 65], fill=(100, 140, 200))
        d.polygon([(20,20),(10,5),(30,15)], fill=(80,120,180))
        d.polygon([(60,20),(70,5),(50,15)], fill=(80,120,180))
    else:
        d.rectangle([15, 20, 65, 65], fill=(200, 130, 80))
        d.rectangle([10, 18, 25, 35], fill=(170,100,60))
        d.rectangle([55, 18, 70, 35], fill=(170,100,60))
    noise = np.random.randint(-12, 12, (80,80,3))
    arr = np.clip(np.array(img).astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def simple_feature(img):
    """feature אחד בלבד: כחוליות ממוצעת"""
    arr = np.array(img.resize((20,20))).astype(float)
    return arr[:,:,2].mean() / 255.0  # ערוץ כחול

def random_predict(_img):
    return random.choice(["חתול 🐱", "כלב 🐶"]), random.uniform(0.45, 0.65)

def smart_predict(img):
    blue = simple_feature(img)
    if blue > 0.45:
        return "חתול 🐱", min(0.95, 0.5 + blue)
    else:
        return "כלב 🐶", min(0.95, 1.0 - blue)

# ═══════════════════════════════════════════════════════════
#  כותרת
# ═══════════════════════════════════════════════════════════
st.markdown("# 🧠 איך מחשב לומד לזהות תמונות?")
st.markdown("מדריך למתחילים – **ללא נוסחאות, רק הבנה!**")

st.markdown("---")

# ═══════════════════════════════════════════════════════════
#  שלב 1 – מה זה דאטה?
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>1</span> הדאטה – התמונות שממנו המחשב לומד", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("המחשב צריך **דוגמאות עם תשובות** כדי ללמוד. כל דוגמה = תמונה + תווית (חתול/כלב).")
    st.markdown('<div class="tip">💡 זה בדיוק כמו ספר תרגילים עם תשובות בסוף – המחשב לומד מהן.</div>', unsafe_allow_html=True)

    cats = [make_image("cat", i) for i in range(5)]
    dogs = [make_image("dog", i+10) for i in range(5)]

    st.markdown("**חתולים 🐱** (תווית: 0)")
    cols = st.columns(5)
    for col, img in zip(cols, cats):
        col.image(img, use_container_width=True)

    st.markdown("**כלבים 🐶** (תווית: 1)")
    cols = st.columns(5)
    for col, img in zip(cols, dogs):
        col.image(img, use_container_width=True)

    st.markdown("**הקוד – כך נראה הדאטה בפייתון:**")
    st.markdown("""<div class="code-area">תמונות  = [🐱, 🐱, 🐶, 🐱, 🐶, ...]   # רשימת תמונות
תוויות  = [ 0,  0,  1,  0,  1, ...]   # 0 = חתול, 1 = כלב

# כל תמונה היא בעצם טבלה של מספרים (פיקסלים)
# תמונה של 80×80 = 6,400 פיקסלים × 3 צבעים = 19,200 מספרים!</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  שלב 2 – Train / Test
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>2</span> מחלקים לאימון ובדיקה", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("לא מראים למחשב את כל התמונות – שומרים חלק כ"בחינה" שהוא לא ראה.")

    total = 10
    test_pct = st.slider("כמה % לבדיקה?", 10, 40, 20, 10)
    n_test = int(total * test_pct / 100)
    n_train = total - n_test

    # פס חזותי פשוט
    fig, ax = plt.subplots(figsize=(7, 0.8), facecolor="#f7f5f0")
    ax.barh(0, n_train, color="#1a1a1a", height=0.5)
    ax.barh(0, n_test, left=n_train, color="#f97316", height=0.5)
    ax.text(n_train/2, 0, f"אימון: {n_train}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(n_train + n_test/2, 0, f"בדיקה: {n_test}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, total); ax.axis("off")
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="tip">💡 הבדיקה היא כמו בחינה – המחשב לא ראה את השאלות מראש!</div>', unsafe_allow_html=True)

    st.markdown("""<div class="code-area">from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    תמונות, תוויות,
    test_size=0.2   # 20% לבדיקה
)</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  שלב 3 – Baseline: ניחוש אקראי
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>3</span> נקודת הפתיחה: ניחוש אקראי (~50%)", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("לפני שבונים מודל חכם – נראה כמה דיוק מקבלים **בלי להשקיע כלום**.")

    if st.button("🎲 הנחש 10 תמונות אקראית"):
        test_imgs = [make_image(random.choice(["cat","dog"]), i+50) for i in range(10)]
        true_labels = [0 if simple_feature(img) > 0.45 else 1 for img in test_imgs]
        correct = 0
        cols = st.columns(5)
        for i, (img, lbl) in enumerate(zip(test_imgs[:5], true_labels[:5])):
            pred, _ = random_predict(img)
            true_name = "חתול 🐱" if lbl == 0 else "כלב 🐶"
            is_ok = (lbl == 0 and "חתול" in pred) or (lbl == 1 and "כלב" in pred)
            if is_ok: correct += 1
            cols[i].image(img, use_container_width=True)
            cols[i].markdown(f"{'✅' if is_ok else '❌'} {pred}")

        acc = correct / 5 * 100
        st.markdown(f"### דיוק: **{acc:.0f}%** מתוך 5 תמונות")
        st.markdown('<div class="tip">💡 ניחוש אקראי נותן בממוצע 50%. כל מודל שלנו חייב לעשות יותר מזה!</div>', unsafe_allow_html=True)

    st.markdown("""<div class="code-area">import random

def random_predict(image):
    return random.choice(["חתול", "כלב"])  # פשוט מנחש!

# דיוק צפוי: 50% (כי שתי מחלקות שוות)</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  שלב 4 – מודל חכם
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>4</span> מודל חכם – לומד מהדאטה", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("המודל שלנו לומד **feature אחד פשוט**: כמה כחול יש בתמונה?")
    st.markdown("חתולים בדאטה שלנו = **כחולים יותר**  |  כלבים = **חומים יותר**")

    st.markdown('<div class="tip">💡 בCNN אמיתי המחשב לומד אלפי features לבד – אנחנו כאן לומדים אחד ידנית כדי להבין את הרעיון.</div>', unsafe_allow_html=True)

    # ויזואליזציה של הfeature
    cat_blues = [simple_feature(make_image("cat", i)) for i in range(20)]
    dog_blues = [simple_feature(make_image("dog", i)) for i in range(20)]

    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="#f7f5f0")
    ax.scatter(cat_blues, [1]*20, color="#3b82f6", s=80, label="חתולים 🐱", zorder=3)
    ax.scatter(dog_blues, [0]*20, color="#f97316", s=80, label="כלבים 🐶", zorder=3)
    ax.axvline(x=0.45, color="#1a1a1a", linewidth=2, linestyle="--", label="גבול ההחלטה")
    ax.set_xlabel("כמה כחול? (0 = אין, 1 = הרבה)", fontsize=10)
    ax.set_yticks([]); ax.set_facecolor("white")
    ax.legend(loc="center right", fontsize=9)
    ax.spines[["top","right","left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("""<div class="code-area">def חשב_feature(תמונה):
    # מחשב את הממוצע של הצבע הכחול בתמונה
    return תמונה[:, :, 2].mean() / 255  # מספר בין 0 ל-1

def חזה(תמונה):
    כחוליות = חשב_feature(תמונה)
    if כחוליות > 0.45:
        return "חתול"   # כחול = חתול
    else:
        return "כלב"    # חום = כלב</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  שלב 5 – נסה בעצמך
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>5</span> נסה בעצמך – מי מנחש טוב יותר?", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.radio("בחר תמונה:", ["חתול 🐱", "כלב 🐶", "הפתע אותי! 🎲"])
        if "הפתע" in chosen:
            kind = random.choice(["cat", "dog"])
        else:
            kind = "cat" if "חתול" in chosen else "dog"
        test_img = make_image(kind, seed=random.randint(0, 999))
        st.image(test_img, width=120)

    with col2:
        true_answer = "חתול 🐱" if kind == "cat" else "כלב 🐶"
        rand_pred, rand_conf = random_predict(test_img)
        smart_pred, smart_conf = smart_predict(test_img)

        st.markdown(f"**תשובה נכונה:** {true_answer}")
        st.markdown("---")

        r_ok = "חתול" in rand_pred and kind=="cat" or "כלב" in rand_pred and kind=="dog"
        s_ok = "חתול" in smart_pred and kind=="cat" or "כלב" in smart_pred and kind=="dog"

        c1, c2 = st.columns(2)
        with c1:
            box = "result-box" if r_ok else "wrong-box"
            st.markdown(f'<div class="{box}">🎲 אקראי<br>{rand_pred}<br><small>{rand_conf:.0%} ביטחון</small></div>', unsafe_allow_html=True)
        with c2:
            box = "result-box" if s_ok else "wrong-box"
            st.markdown(f'<div class="{box}">🧠 חכם<br>{smart_pred}<br><small>{smart_conf:.0%} ביטחון</small></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  שלב 6 – מה זה CNN?
# ═══════════════════════════════════════════════════════════
st.markdown("## <span class='step-number'>6</span> בעולם האמיתי – מה זה CNN?", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
המודל שלנו למד **feature אחד** (כחוליות). CNN אמיתי לומד **אלפי features** לבד:

| מה המחשב שלנו עשה | מה CNN עושה |
|---|---|
| feature אחד בידי אדם | אלפי features אוטומטיים |
| כלל פשוט (כחול → חתול) | כלים מורכבים שהמחשב מצא |
| ~70% דיוק | ~95% דיוק |
    """)

    st.markdown('<div class="tip">💡 CNN = Convolutional Neural Network. בגדול: המחשב מחפש דפוסים (קצוות, צורות, אוזניים...) בשכבות, כשכל שכבה בונה על הקודמת.</div>', unsafe_allow_html=True)

    st.markdown("""<div class="code-area"># CNN בקוד – רק כדי לראות איך זה נראה (לא צריך להבין הכל!)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),  # חיפוש קצוות
    tf.keras.layers.MaxPooling2D(),                    # דחיסת מידע
    tf.keras.layers.Conv2D(64, 3, activation='relu'),  # חיפוש צורות
    tf.keras.layers.Flatten(),                         # ישור לרשימה
    tf.keras.layers.Dense(1, activation='sigmoid'),    # החלטה סופית
])

model.fit(X_train, y_train, epochs=10)  # לימוד!
accuracy = model.evaluate(X_test, y_test)  # בדיקה!</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  סיכום
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🎯 מה למדנו?")
st.markdown("""
1. **דאטה** – המחשב לומד מדוגמאות עם תשובות  
2. **Train/Test** – חלק לאימון, חלק לבדיקה  
3. **Baseline** – ניחוש אקראי = 50%, זה הרף התחתון  
4. **Feature** – מאפיין שעוזר להבדיל בין מחלקות  
5. **CNN** – מוצא features לבד, מדויק הרבה יותר  
""")
