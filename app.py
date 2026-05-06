import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Streamlit Visualization Guide",
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

# ──────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .section-box {
        background: #f8fafc;
        border-left: 5px solid #1a56db;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.2rem;
    }
    .tip-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #14532d;
        margin-top: 0.8rem;
    }
    .tag {
        display: inline-block;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .tag-discrete   { background: #fef3c7; color: #92400e; }
    .tag-continuous { background: #dbeafe; color: #1e40af; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    section = st.radio("Choose a topic:", [
        "1. The Dataset",
        "2. Discrete vs. Continuous",
        "3. Bar Chart",
        "4. Line Chart",
        "5. Scatter Chart",
        "6. Histogram",
    ])
    st.markdown("---")
    st.markdown("**Goal:** predict `test_normal`\n\n`1` = normal result\n\n`0` = abnormal result")

# ══════════════════════════════════════════════
# SECTION 1 – Dataset
# ══════════════════════════════════════════════
if section == "1. The Dataset":
    st.header("The Dataset")

    st.markdown("""
<div class="section-box">
Before drawing charts, let's understand what we have.<br>
The dataset has <strong>100 rows</strong> — each row is one patient.
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients",  len(df))
    col2.metric("Columns",   len(df.columns))
    col3.metric("Normal",    int(df["test_normal"].sum()))
    col4.metric("Abnormal",  int((df["test_normal"] == 0).sum()))

    st.markdown("### First 5 rows")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Column descriptions")
    info = pd.DataFrame({
        "Column":  ["heart_rate", "body_temp", "protein_level", "white_blood_cells", "hemoglobin", "age", "gender", "test_normal"],
        "Meaning": ["Heart rate (bpm)", "Body temperature (C)", "Protein level (g/dL)", "White blood cell count", "Hemoglobin (g/dL)", "Age (years)", "Gender (0 / 1)", "Test result — TARGET"],
        "Type":    ["Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Discrete", "Discrete", "Discrete"],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("### How to generate this dataset")
    st.code("""import pandas as pd
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
elif section == "2. Discrete vs. Continuous":
    st.header("Discrete vs. Continuous Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="tag tag-discrete">Discrete</span>', unsafe_allow_html=True)
        st.markdown("""
A variable that can only take a **limited set of fixed values**.

**Examples in our dataset:**
- `test_normal` — only `0` or `1`
- `gender` — only `0` or `1`
- `age` — whole numbers only

**Question we ask:**
> "How many patients in each category?"

**Best chart: Bar Chart**
""")

    with col2:
        st.markdown('<span class="tag tag-continuous">Continuous</span>', unsafe_allow_html=True)
        st.markdown("""
A variable that can take **any value** within a range.

**Examples in our dataset:**
- `heart_rate` — 60.0, 72.5, 95.3 ...
- `body_temp` — 36.2, 37.1, 38.4 ...
- `hemoglobin` — 11.5, 14.2 ...

**Question we ask:**
> "What is the distribution? Is there a difference between groups?"

**Best chart: Histogram, Scatter**
""")

    st.markdown("---")
    st.markdown("### Quick comparison")
    compare = pd.DataFrame({
        "Property":   ["Possible values", "Example",           "Best chart",        "Typical question"],
        "Discrete":   ["Few, fixed",      "gender: 0 or 1",    "Bar Chart",         "How many in each group?"],
        "Continuous": ["Infinite range",  "heart_rate: 72.3",  "Histogram/Scatter", "What is the range? Any difference?"],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="tip-box">
Tip: if you can count all the possible values on one hand, it is probably discrete.
If there are many different values (like heart rate), it is continuous.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 3 – Bar Chart
# ══════════════════════════════════════════════
elif section == "3. Bar Chart":
    st.header("Bar Chart")

    st.markdown("""
<div class="section-box">
<span class="tag tag-discrete">Best for: Discrete data</span><br><br>
A bar chart shows the <strong>count</strong> of each category.
Use it to answer: <em>"How many patients are normal vs. abnormal?"</em>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Example — count of each test result")

    counts = df["test_normal"].value_counts()
    st.bar_chart(counts)

    st.code("""# Count how many patients in each result category
counts = df["test_normal"].value_counts()

st.bar_chart(counts)
""", language="python")

    st.markdown("""
<div class="tip-box">
What to look for: are the two bars roughly equal, or is the dataset imbalanced?
An imbalanced dataset can make a model look accurate even when it is not.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 4 – Line Chart
# ══════════════════════════════════════════════
elif section == "4. Line Chart":
    st.header("Line Chart")

    st.markdown("""
<div class="section-box">
<span class="tag tag-continuous">Best for: Ordered / sequential data</span><br><br>
A line chart shows how a value <strong>changes along an ordered axis</strong>.
Use it when the order of the X axis matters — like age or time.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Example — heart_rate for each patient, sorted by age")

    df_sorted = df.sort_values("age")
    st.line_chart(df_sorted[["age", "heart_rate"]].set_index("age"))

    st.code("""# Sort by age, then plot heart_rate along that axis
df_sorted = df.sort_values("age")

st.line_chart(df_sorted[["age", "heart_rate"]].set_index("age"))
""", language="python")

    st.markdown("""
<div class="tip-box">
When to use a line chart: when the X axis is ordered and sequential — like age or time.
If X is just categories with no natural order, use a bar chart instead.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 5 – Scatter Chart
# ══════════════════════════════════════════════
elif section == "5. Scatter Chart":
    st.header("Scatter Chart")

    st.markdown("""
<div class="section-box">
<span class="tag tag-continuous">Best for: Continuous data</span><br><br>
A scatter chart shows the <strong>relationship between two variables</strong>.
Each dot is one patient. We split by <code>test_normal</code> so each group gets its own color —
this lets us see whether the two features together can separate normal from abnormal patients.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X axis:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])
    with col2:
        y_col = st.selectbox("Y axis:", ["hemoglobin", "heart_rate", "body_temp", "protein_level", "white_blood_cells"])

    st.markdown(f"### {x_col} vs. {y_col} — colored by test_normal")

    plot_df = pd.DataFrame({
        "x":        df[x_col],
        "normal":   df[y_col].where(df["test_normal"] == 1),
        "abnormal": df[y_col].where(df["test_normal"] == 0),
    }).set_index("x")

    st.scatter_chart(plot_df)

    st.code(f"""# Split by test_normal so each group gets its own color
plot_df = pd.DataFrame({{
    "x":        df["{x_col}"],
    "normal":   df["{y_col}"].where(df["test_normal"] == 1),
    "abnormal": df["{y_col}"].where(df["test_normal"] == 0),
}}).set_index("x")

st.scatter_chart(plot_df)
""", language="python")

    st.markdown("""
<div class="tip-box">
What to look for: if the two colors cluster in different areas of the chart,
those features are strong predictors of test_normal.
Try swapping the axes to find the best separating pair.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 6 – Histogram
# ══════════════════════════════════════════════
elif section == "6. Histogram":
    st.header("Histogram")

    st.markdown("""
<div class="section-box">
<span class="tag tag-continuous">Best for: Continuous data</span><br><br>
A histogram splits values into <strong>bins</strong> and counts how many fall in each bin.
We plot normal and abnormal patients side by side to see whether they have different value ranges.
</div>
""", unsafe_allow_html=True)

    feature = st.selectbox("Choose a feature:", ["heart_rate", "body_temp", "protein_level", "hemoglobin", "white_blood_cells"])
    bins    = st.slider("Number of bins:", 5, 30, 15)

    st.markdown(f"### Distribution of {feature} — normal vs. abnormal")

    edges = np.linspace(df[feature].min(), df[feature].max(), bins + 1)

    normal_counts,   _ = np.histogram(df[df["test_normal"] == 1][feature], bins=edges)
    abnormal_counts, _ = np.histogram(df[df["test_normal"] == 0][feature], bins=edges)

    hist_df = pd.DataFrame(
        {"normal": normal_counts, "abnormal": abnormal_counts},
        index=[f"{e:.1f}" for e in edges[:-1]]
    )
    st.bar_chart(hist_df)

    st.code(f"""import numpy as np

feature = "{feature}"
bins    = {bins}

edges = np.linspace(df[feature].min(), df[feature].max(), bins + 1)

normal_counts,   _ = np.histogram(df[df["test_normal"] == 1][feature], bins=edges)
abnormal_counts, _ = np.histogram(df[df["test_normal"] == 0][feature], bins=edges)

hist_df = pd.DataFrame(
    {{"normal": normal_counts, "abnormal": abnormal_counts}},
    index=[f"{{e:.1f}}" for e in edges[:-1]]
)
st.bar_chart(hist_df)
""", language="python")

    st.markdown("""
<div class="tip-box">
What to look for: if the two colors overlap a lot, this feature does not separate the groups well.
If they sit in different regions, this feature is a strong predictor of test_normal.
Try switching features to find which one separates best.
</div>
""", unsafe_allow_html=True)
