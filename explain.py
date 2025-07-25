import shap
import pandas as pd
import matplotlib.pyplot as plt
from models.xgboost_model import XGBoostModel
from helpers import preprocess_appointments

df = pd.read_csv("data/appointments.csv")
X, y = preprocess_appointments(df)

model = XGBoostModel()
model.load_model("models/xgb_model.pkl")

explainer = shap.Explainer(model.model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("results/shap_summary.png")

shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("results/shap_waterfall_0.png")