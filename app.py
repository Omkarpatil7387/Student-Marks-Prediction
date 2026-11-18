
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Marks Prediction System",
    page_icon="📚",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}

.main .block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 35px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #e5f1f8;
    font-size: 16px;
}

.section-title {
    color: #153e5c;
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
}

.info-card {
    background: white;
    border: 1px solid #e1e9f0;
    padding: 20px;
    border-radius: 12px;
    min-height: 135px;
}

.info-card h3 {
    color: #176b87;
    font-size: 18px;
}

.info-card p {
    color: #526477;
    font-size: 14px;
}

.result-card {
    background: #e9f5f8;
    border: 1px solid #b8dce5;
    padding: 25px;
    border-radius: 14px;
    text-align: center;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_marks.csv"
MODEL_PATH = BASE_DIR / "model" / "student_marks_model.pkl"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Test_Score"
]

TARGET = "Final_Marks"

# ---------------- LOAD OR TRAIN MODEL ----------------

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Please check the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Student Marks Prediction System</h1>
    <p>
        A Machine Learning Based Academic Performance
        Prediction Application
    </p>
    <p>
        Analyze academic performance and estimate expected
        final marks using student-related academic factors.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------

page = st.radio(
    "Navigate",
    ["Home", "Predict Marks", "About Project"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown(
        '<div class="section-title">Welcome to the Prediction System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application uses Machine Learning to estimate "
        "a student's expected final marks based on academic "
        "performance and study-related information."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Academic Analysis</h3>
            <p>Uses study hours, attendance and previous academic scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Marks Prediction</h3>
            <p>Estimates expected final marks using a trained ML model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>Study Guidance</h3>
            <p>Provides general suggestions to support better study habits.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Factors Used for Prediction</div>',
        unsafe_allow_html=True
    )

    factors = [
        ("Study Hours", "Daily time spent studying."),
        ("Attendance", "Student attendance percentage."),
        ("Previous Marks", "Marks obtained in previous examinations."),
        ("Assignment Score", "Performance in assignments."),
        ("Internal Test Score", "Marks obtained in internal tests.")
    ]

    for name, description in factors:
        with st.expander(name):
            st.write(description)

    st.info(
        "Select 'Predict Marks' above to enter academic details "
        "and generate a prediction."
    )

# ---------------- PREDICTION PAGE ----------------

elif page == "Predict Marks":

    st.markdown(
        '<div class="section-title">Student Information</div>',
        unsafe_allow_html=True
    )

    st.write("Enter the student's academic information below.")

    student_name = st.text_input(
        "Student Name (optional)",
        placeholder="Enter student name"
    )

    student_class = st.text_input(
        "Class / Division (optional)",
        placeholder="Example: Third Year - Computer Engineering"
    )

    st.markdown(
        '<div class="section-title">Academic Performance Details</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=0,
            max_value=100,
            value=75
        )

        previous_marks = st.number_input(
            "Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    with col2:
        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        internal_score = st.number_input(
            "Internal Test Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    st.write("")

    if st.button("Predict Final Marks", type="primary", use_container_width=True):

        try:
            model = load_model()

            input_data = pd.DataFrame([{
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Previous_Marks": previous_marks,
                "Assignment_Score": assignment_score,
                "Internal_Test_Score": internal_score
            }])

            prediction = float(model.predict(input_data)[0])
            prediction = float(np.clip(prediction, 0, 100))

            st.divider()

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            if student_name.strip():
                st.write(f"**Student:** {student_name}")

            if student_class.strip():
                st.write(f"**Class:** {student_class}")

            st.markdown(f"""
            <div class="result-card">
                <p style="color:#526477; font-size:17px;">
                    Estimated Final Marks
                </p>
                <h1 style="color:#176b87; font-size:48px;">
                    {prediction:.2f} / 100
                </h1>
                <p style="color:#526477;">
                    Machine Learning Prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(prediction / 100)

            if prediction >= 75:
                st.success("The model estimates a strong academic performance.")
            elif prediction >= 50:
                st.info("The model estimates a moderate academic performance.")
            else:
                st.warning(
                    "The model estimates that additional academic support "
                    "and study planning may be useful."
                )

            st.markdown("### Study Improvement Suggestions")

            suggestions = []

            if study_hours < 2:
                suggestions.append(
                    "Create a consistent daily study routine."
                )

            if attendance < 75:
                suggestions.append(
                    "Try to attend classes regularly and revise missed topics."
                )

            if assignment_score < 60:
                suggestions.append(
                    "Spend more time understanding and completing assignments."
                )

            if internal_score < 60:
                suggestions.append(
                    "Practice questions and revise internal test topics."
                )

            if previous_marks < 60:
                suggestions.append(
                    "Review previous mistakes and strengthen basic concepts."
                )

            if not suggestions:
                suggestions.append(
                    "Continue your current study habits and revise regularly."
                )

            for suggestion in suggestions:
                st.write(f"- {suggestion}")

            st.caption(
                "Note: This result is an estimate from a practice ML model, "
                "not a guaranteed examination result."
            )

        except Exception as e:
            st.error(f"Unable to generate prediction: {e}")

# ---------------- ABOUT PAGE ----------------

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.write("""
    The Student Marks Prediction System is a Machine Learning
    project developed to estimate student academic performance.

    The system uses academic information such as study hours,
    attendance, previous marks, assignment scores and internal
    test scores to predict expected final marks.
    """)

    st.markdown("### Project Objectives")

    st.write("""
    - Understand student academic performance.
    - Apply regression algorithms to a marks prediction problem.
    - Train and evaluate Machine Learning models.
    - Build a simple web application for predictions.
    - Present prediction results in an understandable format.
    """)

    st.markdown("### How the System Works")

    steps = [
        "Collect student academic information.",
        "Prepare and process the dataset.",
        "Train regression models using historical practice data.",
        "Select a trained model for prediction.",
        "Enter new student information.",
        "Display estimated final marks."
    ]

    for i, step in enumerate(steps, 1):
        st.write(f"**Step {i}:** {step}")

    st.markdown("### Technology Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        <div class="info-card">
            <h3>Python</h3>
            <p>Programming language used for model development.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
        <div class="info-card">
            <h3>Scikit-learn</h3>
            <p>Machine Learning algorithms and model training.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
        <div class="info-card">
            <h3>Streamlit</h3>
            <p>Interactive web application development.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Dataset Information")

    st.write("""
    This project package uses a synthetically generated practice
    dataset. It is intended to demonstrate the Machine Learning
    workflow and should not be treated as verified real-world
    student performance data.
    """)

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#718096; padding:15px;">
    <strong>Student Marks Prediction System</strong>
    <br>
    Machine Learning Academic Project
    <br>
    <small>Developed using Python, Scikit-learn and Streamlit</small>
</div>
""", unsafe_allow_html=True)
