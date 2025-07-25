import gradio as gr
import pandas as pd
from models.xgboost_model import XGBoostModel
from helpers import preprocess_appointments

model = XGBoostModel()
model.load_model("models/xgb_model.pkl")

def predict_manually(scheduled_days_ahead, appointment_dayofweek, is_weekend, sex, age_group):
    sex_encoded = 0 if sex == "F" else 1
    age_group_encoded = {"0-18": 0, "19-35": 1, "36-60": 2, "60+": 3}[age_group]
    data = pd.DataFrame([[
        scheduled_days_ahead, appointment_dayofweek, is_weekend,
        sex_encoded, age_group_encoded
    ]], columns=["scheduled_days_ahead", "appointment_dayofweek", "is_weekend", "sex_encoded", "age_group_encoded"])
    prediction = model.predict(data)[0]
    return "No-Show" if prediction == 1 else "Attended"

def predict_from_csv(file):
    df = pd.read_csv(file.name)
    X, _ = preprocess_appointments(df)
    preds = model.predict(X)
    df["prediction"] = ["No-Show" if p == 1 else "Attended" for p in preds]
    return df

with gr.Blocks() as demo:
    with gr.Tab("Manual Prediction"):
        scheduled_days_ahead = gr.Slider(0, 100, step=1, label="Scheduled Days Ahead", value=5)
        appointment_dayofweek = gr.Slider(0, 6, step=1, label="Appointment Day of Week", value=2)
        is_weekend = gr.Radio([0, 1], label="Is Weekend?", value=0)
        sex = gr.Radio(["M", "F"], label="Sex", value="F")
        age_group = gr.Dropdown(["0-18", "19-35", "36-60", "60+"], label="Age Group", value="19-35")
        result_output = gr.Textbox(label="Prediction")
        predict_btn = gr.Button("Predict")
        predict_btn.click(
            predict_manually,
            inputs=[scheduled_days_ahead, appointment_dayofweek, is_weekend, sex, age_group],
            outputs=result_output
        )

    with gr.Tab("CSV Batch Prediction"):
        file_input = gr.File(label="Upload CSV File", type="filepath")
        batch_output = gr.Dataframe(label="Predictions", interactive=False)
        file_input.change(
            predict_from_csv,
            inputs=file_input,
            outputs=batch_output
        )

demo.launch(share=True)