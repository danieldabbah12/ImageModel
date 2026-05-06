import streamlit as st
import pandas as pd
import numpy as np

# הפונקציה ליצירת הדאטה-סט (כפי שסיפקת)
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
    
    # רעש
    noise_size = int(n * noise_ratio)
    noise_idx = np.random.choice(n, noise_size, replace=False)
    df.loc[noise_idx, "test_normal"] = 1 - df.loc[noise_idx, "test_normal"]
    
    # עיגול
    df["heart_rate"] = df["heart_rate"].round(0)
    df["body_temp"] = df["body_temp"].round(1)
    df["protein_level"] = df["protein_level"].round(2)
    df["white_blood_cells"] = df["white_blood_cells"].round(0)
    df["hemoglobin"] = df["hemoglobin"].round(1)
    
    return df

# הגדרת הגדרות עמוד
st.set_page_config(page_title="הדגמת גרפים ב-Streamlit", layout="wide")

st.title("📊 אפליקציית בדיקות דם - הדגמת `st.bar_chart`")
st.write("אפליקציה זו מציגה כיצד להשתמש בפונקציית `st.bar_chart` של Streamlit על בסיס נתוני בדיקות דם שנוצרו סינתטית.")

# טעינת הנתונים
df = generate_blood_test_dataset()

# הוספת עמודות טקסטואליות כדי שהגרפים יהיו ברורים יותר
df["Test_Result"] = df["test_normal"].map({1: "תקין (Normal)", 0: "חריג (Abnormal)"})
df["Gender_Label"] = df["gender"].map({0: "זכר", 1: "נקבה"})

st.subheader("הצצת נתונים (Data Preview)")
st.dataframe(df.head(10))

st.divider()

# ==========================================
# דוגמה 1: תרשים עמודות בסיסי (Basic Bar Chart)
# ==========================================
st.subheader("1. תרשים עמודות פשוט - ממוצע מדדים")
st.write("מציג את ממוצע דופק הלב (Heart Rate) וההמוגלובין לפי תוצאת הבדיקה.")

# הכנת נתונים ממוצעים לגרף
df_mean = df.groupby("Test_Result")[["heart_rate", "hemoglobin"]].mean().reset_index()

st.bar_chart(
    df_mean, 
    x="Test_Result", 
    y=["heart_rate", "hemoglobin"]
)

# ==========================================
# דוגמה 2: תרשים עמודות עם צבעים (Colored Bar Chart)
# ==========================================
st.subheader("2. תרשים עם הפרדת צבעים (Color Parameter)")
st.write("מציג את ממוצע כדוריות הדם הלבנות (WBC) לפי תוצאת הבדיקה, צבוע לפי מגדר.")

# קיבוץ לפי תוצאה ומגדר
df_grouped_wbc = df.groupby(["Test_Result", "Gender_Label"])["white_blood_cells"].mean().reset_index()

st.bar_chart(
    df_grouped_wbc, 
    x="Test_Result", 
    y="white_blood_cells", 
    color="Gender_Label"
)

# ==========================================
# דוגמה 3: תרשים עמודות אופקי (Horizontal Bar Chart)
# ==========================================
st.subheader("3. תרשים עמודות אופקי (Horizontal)")
st.write("שימוש בפרמטר `horizontal=True` כדי להפוך את כיוון העמודות.")

st.bar_chart(
    df_mean, 
    x="Test_Result", 
    y="heart_rate", 
    horizontal=True,
    color="#FF4B4B" # ניתן גם להעביר צבע ספציפי בפורמט HEX
)

# ==========================================
# דוגמה 4: תרשים עמודות לא מוערם (Unstacked Bar Chart)
# ==========================================
st.subheader("4. תרשים עמודות מפוצל - Unstacked")
st.write("שימוש בפרמטר `stack=False` כדי להציג את העמודות אחת ליד השנייה במקום אחת על השנייה.")

df_grouped_temp = df.groupby(["Test_Result", "Gender_Label"])["body_temp"].mean().reset_index()

st.bar_chart(
    df_grouped_temp, 
    x="Test_Result", 
    y="body_temp", 
    color="Gender_Label", 
    stack=False
)
