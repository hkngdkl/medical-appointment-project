import pandas as pd

df = pd.read_csv("data/appointments.csv")
print("Status dağılımı:")
print(df["status"].value_counts())