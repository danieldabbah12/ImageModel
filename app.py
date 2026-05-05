import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import random

st.set_page_config(page_title="סיווג תמונות – למתחילים", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap');
* { font-family: 'Heebo', sans-serif; direction: rtl; }
body, .stApp { background: #f7f5f0; color: #1a1a1a; }
h1 { font-size: 2rem; font-weight: 900; }
h2 { font-size: 1.3rem; font-weight: 700; border-bottom: 3px solid #1a1a1a; padding-bottom: 6px; }
.card { background: white; border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; border: 1.5px solid #e0ddd8; box-shadow: 2px 3px 0px #d0ccc4; }
.step { display: inline-block; width: 30px; height: 30px; background: #1a1a1a; color: white; border-radius: 50%; font-weight: 900; text-align: center; line-height: 30px; margin-left: 8px; }
.tip { background: #fffbeb; border-right: 4px solid #f59e0b; padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; font-size: 0.9rem; margin: 0.8rem 0; }
.code-area { background: #1a1a1a; color: #a8ff78; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.82rem; line-height: 1.7; white-space: pre; overflow-x: auto; margin-top: 0.8rem; }
.result-ok  { background: #f0fdf4; border: 2px solid #22c55e; border-radius: 10px; padding: 1rem; text-align: center; font-size: 1.3rem; font-weight: 900; }
.result-bad { background: #fff7ed; border: 2px solid #f97316; border-radius: 10px; padding: 1rem; text-align: center; font-size: 1.3rem; font-weight: 900; }
.bar-wrap { background: #e5e7eb; border-radius: 6px; height: 28px; overflow: hidden; display: flex; margin: 0.5rem 0; }
.bar-train { background: #1a1a1a; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.85rem; }
.bar-test  { background: #f97316; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.85rem; }
.dot-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
.dot { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── פונקציות עזר ────────────────────────────────────────────

def make_image(kind, seed=0):
    """תמונה פשוטה: עיגול כחול = חתול | מלבן חום = כלב"""
    random.seed(seed); np.random.seed(seed)
    img = Image.new("RGB", (80, 80), (230, 225, 215))
    d = ImageDraw.Draw(img)
    if kind == "cat":
        d.ellipse([14, 14, 66, 66], fill=(90, 130, 200))
        d.polygon([(20,22),(10,4),(30,16)], fill=(70,110,180))
        d.polygon([(60,22),(70,4),(50,16)], fill=(70,110,180))
    else:
        d.rectangle([14, 20, 66, 66], fill=(200, 128, 75))
        d.rectangle([ 8, 16, 24, 34], fill=(170, 100, 55))
        d.rectangle([56, 16, 72, 34], fill=(170, 100, 55))
    noise = np.random.randint(-10, 10, (80, 80, 3))
    arr = np.clip(np.array(img).astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def blue_feature(img):
    """feature יחיד: ממוצע ערוץ כחול"""
    arr = np.array(img.resize((20, 20))).astype(float)
    return arr[:, :, 2].mean() / 255.0

def random_predict(_img):
    pred = random.choice(["חתול 🐱", "כלב 🐶"])
    return pred, random.uniform(0.45, 0.60)

def smart_predict(img):
    b = blue_feature(img)
    if b > 0.45:
        return "חתול 🐱", round(min(0.95, 0.5 + b), 2)
    else:
        return "כלב 🐶",  round(min(0.95, 1.1 - b), 2)

def is_correct(pred, kind):
    return ("חתול" in pred and kind == "cat") or ("כלב" in pred and kind == "dog")

# ─── כותרת ───────────────────────────────────────────────────

st.markdown("# 🧠 איך מחשב לומד לזהות תמונות?")
st.markdown("מדריך למתחילים — **ללא נוסחאות, רק הבנה!**")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# שלב 1 – דאטה
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>1</span> הדאטה – תמונות עם תשובות", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("המחשב צריך **דוגמאות עם תשובות** כדי ללמוד. בדיוק כמו ספר תרגילים עם פתרונות.")
st.markdown('<div class="tip">💡 כל תמונה = טבלה של מספרים (פיקסלים). תמונה 80×80 = 19,200 מספרים!</div>', unsafe_allow_html=True)

st.markdown("**חתולים 🐱** — תווית: 0")
cols = st.columns(6)
for i, col in enumerate(cols):
    col.image(make_image("cat", i), use_container_width=True)

st.markdown("**כלבים 🐶** — תווית: 1")
cols = st.columns(6)
for i, col in enumerate(cols):
    col.image(make_image("dog", i+20), use_container_width=True)

st.markdown("""<div class="code-area">תמונות = [🐱, 🐱, 🐶, 🐱, 🐶, ...]
תוויות = [ 0,  0,  1,  0,  1, ...]   # 0=חתול, 1=כלב</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# שלב 2 – Train / Test
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>2</span> מחלקים לאימון ובדיקה", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("שומרים חלק מהתמונות בצד — המחשב **לא רואה אותן** בזמן האימון. הן ישמשו כ'בחינה' בסוף.")

test_pct = st.slider("כמה אחוז לבדיקה?", 10, 40, 20, 10)
train_pct = 100 - test_pct
st.markdown(f"""
<div class="bar-wrap">
  <div class="bar-train" style="width:{train_pct}%">אימון {train_pct}%</div>
  <div class="bar-test"  style="width:{test_pct}%">בדיקה {test_pct}%</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="tip">💡 בדרך כלל: 80% אימון, 20% בדיקה.</div>', unsafe_allow_html=True)
st.markdown("""<div class="code-area">from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    תמונות, תוויות, test_size=0.2
)</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# שלב 3 – Baseline
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>3</span> נקודת פתיחה: ניחוש אקראי (~50%)", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("לפני שבונים מודל חכם — כמה דיוק מקבלים **בלי להשקיע כלום**?")

if st.button("🎲 הנחש 10 תמונות אקראית"):
    kinds = [random.choice(["cat","dog"]) for _ in range(10)]
    imgs  = [make_image(k, i+100) for i, k in enumerate(kinds)]
    preds = [random_predict(img)[0] for img in imgs]
    correct_count = sum(is_correct(p, k) for p, k in zip(preds, kinds))

    cols = st.columns(10)
    for col, img, pred, kind in zip(cols, imgs, preds, kinds):
        col.image(img, use_container_width=True)
        col.markdown("✅" if is_correct(pred, kind) else "❌")

    st.markdown(f"### תוצאה: **{correct_count}/10** נכון = **{correct_count*10}%** דיוק")
    st.markdown('<div class="tip">💡 ניחוש אקראי נותן ~50%. כל מודל חייב לעשות יותר מזה!</div>', unsafe_allow_html=True)

st.markdown("""<div class="code-area">def random_predict(image):
    return random.choice(["חתול", "כלב"])  # מנחש בלי להסתכל!

# דיוק צפוי: ~50% (כי יש 2 מחלקות שוות)</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# שלב 4 – מודל חכם
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>4</span> מודל חכם — לומד מאפיין אחד", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("המודל שלנו לומד **feature אחד בלבד**: כמה **כחול** יש בתמונה?")
st.markdown("חתולים בדאטה שלנו = **כחולים** | כלבים = **חומים**")
st.markdown('<div class="tip">💡 CNN אמיתי לומד אלפי features לבד. כאן אנחנו לומדים אחד ידנית — כדי להבין את הרעיון.</div>', unsafe_allow_html=True)

# ויזואליזציה פשוטה עם HTML במקום matplotlib
cat_blues = [blue_feature(make_image("cat", i)) for i in range(12)]
dog_blues = [blue_feature(make_image("dog", i)) for i in range(12)]

st.markdown("**כמה כחול יש בכל תמונה? (0 = אין, 1 = הרבה)**")

st.markdown("חתולים 🐱")
dots_html = '<div class="dot-row">'
for b in cat_blues:
    pct = int(b * 100)
    dots_html += f'<div class="dot" style="background:rgb(30,{int(80+b*150)},{int(150+b*100)})" title="{pct}%">{pct}</div>'
dots_html += '</div>'
st.markdown(dots_html, unsafe_allow_html=True)

st.markdown("כלבים 🐶")
dots_html = '<div class="dot-row">'
for b in dog_blues:
    pct = int(b * 100)
    dots_html += f'<div class="dot" style="background:rgb({int(160+b*50)},{int(90+b*40)},{int(40+b*80)})" title="{pct}%">{pct}</div>'
dots_html += '</div>'
st.markdown(dots_html, unsafe_allow_html=True)

st.markdown("**גבול ההחלטה: אם כחוליות > 45 → חתול, אחרת → כלב**")

st.markdown("""<div class="code-area">def feature(תמונה):
    # ממוצע ערוץ כחול (0–255), מנורמל ל-0–1
    return תמונה[:, :, 2].mean() / 255

def חזה(תמונה):
    כחוליות = feature(תמונה)
    if כחוליות > 0.45:
        return "חתול"   # כחול = חתול
    else:
        return "כלב"    # חום = כלב</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# שלב 5 – נסה בעצמך
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>5</span> נסה בעצמך — מי מנחש טוב יותר?", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

choice = st.radio("בחר תמונה:", ["חתול 🐱", "כלב 🐶", "הפתע אותי! 🎲"], horizontal=True)
kind = {"חתול 🐱": "cat", "כלב 🐶": "dog"}.get(choice, random.choice(["cat","dog"]))
seed = random.randint(0, 999) if "הפתע" in choice else 77
img = make_image(kind, seed)

col_img, col_res = st.columns([1, 2])
with col_img:
    st.image(img, width=130)
    true_name = "חתול 🐱" if kind == "cat" else "כלב 🐶"
    st.markdown(f"**תשובה נכונה:** {true_name}")

with col_res:
    r_pred, r_conf = random_predict(img)
    s_pred, s_conf = smart_predict(img)
    r_ok = is_correct(r_pred, kind)
    s_ok = is_correct(s_pred, kind)

    c1, c2 = st.columns(2)
    with c1:
        cls = "result-ok" if r_ok else "result-bad"
        st.markdown(f'<div class="{cls}">🎲 אקראי<br>{r_pred}<br><small>ביטחון: {r_conf:.0%}</small></div>', unsafe_allow_html=True)
    with c2:
        cls = "result-ok" if s_ok else "result-bad"
        st.markdown(f'<div class="{cls}">🧠 חכם<br>{s_pred}<br><small>ביטחון: {s_conf:.0%}</small></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# שלב 6 – CNN
# ══════════════════════════════════════════════════════════════
st.markdown("## <span class='step'>6</span> בעולם האמיתי — מה זה CNN?", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""
| המודל שלנו | CNN אמיתי |
|---|---|
| feature אחד בידי אדם | אלפי features אוטומטיים |
| כלל פשוט (כחול → חתול) | שכבות מורכבות שהמחשב מצא |
| ~70% דיוק | ~95% דיוק |
""")
st.markdown('<div class="tip">💡 CNN = Convolutional Neural Network. המחשב מחפש קצוות, צורות, אוזניים... בשכבות. כל שכבה בונה על הקודמת.</div>', unsafe_allow_html=True)

st.markdown("""<div class="code-area">import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),  # מחפש קצוות
    tf.keras.layers.MaxPooling2D(),                    # מדחס מידע
    tf.keras.layers.Conv2D(64, 3, activation='relu'),  # מחפש צורות
    tf.keras.layers.Flatten(),                         # מיישר לרשימה
    tf.keras.layers.Dense(1, activation='sigmoid'),    # מחליט: חתול/כלב
])

model.fit(X_train, y_train, epochs=10)   # לומד!
model.evaluate(X_test, y_test)           # נבחן!</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─── סיכום ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 מה למדנו?")
st.markdown("""
1. **דאטה** — המחשב לומד מדוגמאות עם תשובות  
2. **Train/Test** — חלק לאימון, חלק לבדיקה (כמו בחינה!)  
3. **Baseline** — ניחוש אקראי = ~50%, זה הרף התחתון  
4. **Feature** — מאפיין שעוזר להבדיל בין מחלקות  
5. **CNN** — מוצא features לבד, מדויק הרבה יותר  
""")
