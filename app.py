import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .header {
        background: linear-gradient(135deg, #183b56 0%, #1f6f8b 100%);
        padding: 30px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(24, 59, 86, 0.18);
    }

    .header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .header p {
        margin: 8px 0 0;
        font-size: 15px;
        opacity: 0.92;
    }

    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 26px 18px;
        margin-bottom: 20px;
        box-shadow: 0 5px 18px rgba(30, 60, 90, 0.06);
    }

    .section-title {
        color: #183b56;
        font-size: 20px;
        font-weight: 650;
        margin-bottom: 16px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
    }

    .result-card {
        background: linear-gradient(135deg, #e9f7f3 0%, #f5fbff 100%);
        border: 1px solid #b8e0d2;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 10px;
    }

    .result-label {
        color: #55706a;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .result-value {
        color: #0f766e;
        font-size: 48px;
        font-weight: 750;
        margin: 5px 0;
    }

    .result-sub {
        color: #475569;
        font-size: 14px;
    }

    .info-box {
        background: #eef5fb;
        border-left: 4px solid #1f6f8b;
        padding: 13px 15px;
        border-radius: 8px;
        color: #334155;
        font-size: 13px;
        margin-bottom: 18px;
    }

    div[data-testid="stSidebar"] {
        background: #eef3f7;
    }

    .footer {
        text-align: center;
        color: #8a97a5;
        font-size: 12px;
        padding-top: 12px;
    }

    button[kind="primary"] {
        background: #1f6f8b !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_marks.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_marks_model.pkl")

@st.cache_resource
def load_or_train_model():
    """Load the saved model, or train one automatically on first launch."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    data = pd.read_csv(DATA_PATH)
    features = [
        "Study_Hours", "Attendance", "Previous_Marks",
        "Assignment_Score", "Internal_Test_Score"
    ]
    X = data[features]
    y = data["Final_Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=8, random_state=42
        )
    }
    best_model = None
    best_rmse = float("inf")
    for candidate in candidates.values():
        candidate.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, candidate.predict(X_test)) ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = candidate

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    return best_model

st.sidebar.markdown("### Student Marks Prediction")
st.sidebar.markdown(
    "A simple machine learning application for estimating final marks from academic inputs."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model inputs**")
st.sidebar.write("• Study hours")
st.sidebar.write("• Attendance")
st.sidebar.write("• Previous marks")
st.sidebar.write("• Assignment score")
st.sidebar.write("• Internal test score")
st.sidebar.markdown("---")
st.sidebar.caption("College ML Project")

# ---------- Header ----------
st.markdown("""
<div class="header">
    <h1>Student Marks Prediction</h1>
    <p>Machine Learning based prediction of a student's expected final marks</p>
</div>
""", unsafe_allow_html=True)

# ---------- Main input area ----------
st.markdown("""
<div class="section-card">
    <div class="section-title">Student Information</div>
    <div class="info-box">
        Enter the student's academic details below. The trained regression model
        will estimate the expected final marks out of 100.
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5
    )

    previous_marks = st.number_input(
        "Previous Exam Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    internal_test = st.number_input(
        "Internal Test Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    assignment_score = st.number_input(
        "Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    st.markdown(
        '<div class="small-note">Marks are assumed to be out of 100.</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prediction ----------
if st.button("Predict Final Marks", type="primary", use_container_width=True):
    with st.spinner("Preparing the prediction model..."):
        model = load_or_train_model()

    input_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Test_Score": internal_test
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))

    if prediction >= 75:
        performance = "Strong performance"
    elif prediction >= 60:
        performance = "Good performance"
    elif prediction >= 40:
        performance = "Average performance"
    else:
        performance = "Needs improvement"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Final Marks</div>
        <div class="result-value">{prediction:.1f} / 100</div>
        <div class="result-sub">{performance}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
    Student Marks Prediction | Machine Learning Mini Project
</div>
""", unsafe_allow_html=True)
