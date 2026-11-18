
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Marks Prediction",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

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
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    border: 1px solid #e1e9f0;
    padding: 22px;
    border-radius: 12px;
    min-height: 145px;
}

.info-card h3 {
    color: #176b87;
    font-size: 19px;
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

# ---------------- SESSION STORAGE ----------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError("Student dataset was not found.")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["Final_Marks"]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES]
    y = df["Final_Marks"]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Student Marks Prediction System</h1>
    <p>Estimate academic performance using student study and assessment details.</p>
    <p>Enter academic information, view predicted marks and keep a record of results.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------

page = st.radio(
    "Navigation",
    ["Home", "Predict Marks", "Saved Results"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.markdown(
        '<div class="section-title">Welcome</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Use this application to estimate expected final marks
    based on academic performance and study-related information.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Academic Details</h3>
            <p>Enter study hours, attendance and previous examination marks.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Marks Prediction</h3>
            <p>Get an estimated final score based on the entered information.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>Saved Records</h3>
            <p>View previous predictions made during your current session.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Information Used</div>',
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

    st.info("Open Predict Marks to enter details and calculate an estimate.")

# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Predict Marks":

    st.markdown(
        '<div class="section-title">Student Information</div>',
        unsafe_allow_html=True
    )

    st.write("Enter the details below to calculate estimated final marks.")

    with st.form("prediction_form"):

        student_name = st.text_input(
            "Name (optional)",
            placeholder="Enter name"
        )

        student_class = st.text_input(
            "Class / Division (optional)",
            placeholder="Enter class or division"
        )

        st.markdown("### Academic Performance")

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

        submitted = st.form_submit_button(
            "Predict Final Marks",
            use_container_width=True
        )

    if submitted:

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

            record = {
                "Date and Time": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                ),
                "Student Name": student_name.strip() or "Not provided",
                "Class": student_class.strip() or "Not provided",
                "Study Hours": study_hours,
                "Attendance": attendance,
                "Previous Marks": previous_marks,
                "Assignment Score": assignment_score,
                "Internal Test Score": internal_score,
                "Predicted Marks": round(prediction, 2)
            }

            # Automatically save the result in this session
            st.session_state.prediction_history.append(record)
            st.session_state.last_prediction = record

        except Exception as error:
            st.error(f"Prediction could not be generated: {error}")

    # DISPLAY LATEST RESULT

    result = st.session_state.last_prediction

    if result is not None:

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        if result["Student Name"] != "Not provided":
            st.write(f"**Student:** {result['Student Name']}")

        if result["Class"] != "Not provided":
            st.write(f"**Class:** {result['Class']}")

        st.markdown(f"""
        <div class="result-card">
            <p style="color:#526477; font-size:17px;">
                Estimated Final Marks
            </p>
            <h1 style="color:#176b87; font-size:48px;">
                {result['Predicted Marks']:.2f} / 100
            </h1>
            <p style="color:#526477;">
                Estimated result
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(result["Predicted Marks"] / 100)

        if result["Predicted Marks"] >= 75:
            st.success("The estimated score is in the higher range.")

        elif result["Predicted Marks"] >= 50:
            st.info("The estimated score is in the middle range.")

        else:
            st.warning("The estimated score is in the lower range.")

        st.markdown("### Study Suggestions")

        suggestions = []

        if result["Study Hours"] < 2:
            suggestions.append("Build a regular daily study routine.")

        if result["Attendance"] < 75:
            suggestions.append("Try to attend classes regularly.")

        if result["Assignment Score"] < 60:
            suggestions.append("Give more attention to assignments.")

        if result["Internal Test Score"] < 60:
            suggestions.append("Practice questions and revise internal topics.")

        if result["Previous Marks"] < 60:
            suggestions.append("Revise basic concepts and previous mistakes.")

        if not suggestions:
            suggestions.append(
                "Continue your current study routine and revise regularly."
            )

        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        st.success("This prediction has been added to Saved Results.")

        st.caption(
            "Predictions are estimates based on the available practice dataset "
            "and are not guaranteed examination results."
        )

# ==================================================
# SAVED RESULTS PAGE
# ==================================================

elif page == "Saved Results":

    st.markdown(
        '<div class="section-title">Saved Results</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history

    if len(history) == 0:

        st.info("No saved predictions yet. Go to Predict Marks to create one.")

    else:

        results_df = pd.DataFrame(history)

        total = len(results_df)
        average = results_df["Predicted Marks"].mean()
        highest = results_df["Predicted Marks"].max()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total)
        col2.metric("Average Predicted Marks", f"{average:.2f}")
        col3.metric("Highest Predicted Marks", f"{highest:.2f}")

        st.markdown("### Prediction History")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="student_prediction_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Saved Results", type="secondary"):
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.rerun()

# ---------------- END ----------------
