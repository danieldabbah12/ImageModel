import streamlit as st
import pandas as pd
import numpy as np

# ======================
# פונקציית יצירת דאטה
# ======================
def generate_blood_test_dataset(n=100, noise_ratio=0.08, random_state=42):
    np.random.seed(random_state)
    
    half = n // 2

    normal = pd.DataFrame({
        "heart_rate": np.random.normal(72, 8, half),
        "body_temp": np.random.normal(36.6, 0.3, half),
        "protein_level": np.random.normal(7.0, 0.5, half),
        "wbc": np.random.normal(7000, 1200, half),
        "hemoglobin": np.random.normal(14, 1.5, half),
        "age": np.random.randint(18, 65, half),
        "gender": np.random.randint(0, 2, half),
        "test_normal": 1
    })

    abnormal = pd.DataFrame({
        "heart_rate": np.random.normal(90, 15, half),
        "body_temp": np.random.normal(37.8, 0.8, half),
        "protein_level": np.random.normal(6.0, 1.0, half),
        "wbc": np.random.normal(11000, 3000, half),
        "hemoglobin": np.random.normal(11.5, 2.0, half),
        "age": np.random.randint(18, 80, half),
        "gender": np.random.randint(0, 2, half),
        "test_normal": 0
    })

    df = pd.concat([normal, abnormal]).sample(frac=1).reset_index(drop=True)

    noise_size = int(n * noise_ratio)
    noise_idx = np.random.choice(n, noise_size, replace=False)
    df.loc[noise_idx, "test_normal"] = 1 - df.loc[noise_idx, "test_normal"]

    # עיגול
    df["heart_rate"] = df["heart_rate"].round(0)
    df["body_temp"] = df["body_temp"].round(1)
    df["protein_level"] = df["protein_level"].round(2)
    df["wbc"] = df["wbc"].round(0)
    df["hemoglobin"] = df["hemoglobin"].round(1)

    return df


# ======================
# UI
# ======================
st.title("🩸 בדיקות דם - תקין או לא")

# יצירת דאטה
df = generate_blood_test_dataset()

# הצגת דאטה
st.subheader("הדאטה")
st.dataframe(df)

# ======================
# גרף פשוט
# ======================
st.subheader("חום גוף מול WBC")

st.scatter_chart(
    df,
    x="body_temp",
    y="wbc",
    color="test_normal"
)

# ======================
# התפלגות
# ======================
st.subheader("כמה תקין / לא תקין")
st.bar_chart(df["test_normal"].value_counts())
