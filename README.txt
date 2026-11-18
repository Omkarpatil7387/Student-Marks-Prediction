# Student Marks Prediction — Deployment Package

This project can be deployed as a website using Streamlit Community Cloud.

## Files to upload to GitHub
Upload the CONTENTS of this folder to the top level of a new GitHub repository:
- app.py
- requirements.txt
- data/student_marks.csv
- notebook/Student_Marks_Prediction.ipynb
- train_model.py
- PROJECT_GUIDE.txt

## Streamlit deployment
1. Create a GitHub repository.
2. Upload these project files and folders (not the ZIP itself).
3. Open https://share.streamlit.io and connect the repository.
4. Select `app.py` as the app file and deploy.
5. On first prediction, the app automatically trains and caches a model using the included CSV.

No local Python setup or pre-trained model file is required to deploy the website.

## Local use (optional)
Install requirements with `pip install -r requirements.txt`, then run `streamlit run app.py`.

## Inputs
Study hours, attendance, previous marks, assignment score, internal test score.
Target: final marks out of 100.

## Important dataset note
The included 500-row CSV is synthetic practice data, not real student records or an official UCI/Kaggle dataset. Predictions demonstrate the machine-learning workflow and do not establish real-world predictive accuracy. Disclose this honestly during your presentation.
