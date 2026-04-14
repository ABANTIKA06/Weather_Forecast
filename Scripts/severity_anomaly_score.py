# Import Libraries
from sklearn.ensemble import IsolationForest
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 

#To set maximum width for features while showing the output on terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load final clean dataset
df = pd.read_csv("Data_Raw/final_history.csv")

# Select multivariate features
features = df[[
    "mean_temp",
    "max_temp",
    "min_temp",
    "precipitation",
    "cloud_cover",
    "temp_range",
    "temp_avg_change"
]]

# Create model
model = IsolationForest(contamination=0.1, random_state=1)

# Predict anomalies
df["multi_anomaly"] = model.fit_predict(features)

# Convert output (-1 → 1 anomaly, 0 normal)
df["multi_anomaly"] = np.where(df["multi_anomaly"] == -1, 1, 0)

print(df.head())
print(len(df))

# Cnvert date to datetime because it is in string previously
df["date"] = pd.to_datetime(df["date"])
print(df["date"].dtype)

# Understanding why did anomalies happen
anomaly_days = df[df["multi_anomaly"] == 1]
print(anomaly_days.describe())

normal = df[df["multi_anomaly"] == 0]
anomaly = df[df["multi_anomaly"] == 1]

print("Normal Days:\n", normal.mean(numeric_only=True))
print("Anomaly Days:\n", anomaly.mean(numeric_only=True))

numeric_cols = df.select_dtypes(include=["number"]).columns

difference = anomaly[numeric_cols].mean() - normal[numeric_cols].mean()
print("The difference in between Anomaly Days and Normal Days is:\n",difference)

# Get anomaly score( using decision gives 2 results
# POSITIVE = NORMAL, NGATIVE SCORE = ANOMALY)
df["anomaly_score"] = model.decision_function(features)

# Convert to strength( Higher value equals to stronger anomaly)
df["anomaly_strength"] = -df["anomaly_score"]

df = df.sort_values("date")  # Ensure proper timeline
df["smooth_strength"] = df["anomaly_strength"].rolling(window=7).mean()


# Check for top anomalies
top_anomalies = df.sort_values("anomaly_strength", ascending = False).head(20)

print(top_anomalies[["date","mean_temp", "precipitation", "cloud_cover", "anomaly_strength"]])
print(df["anomaly_strength"].min())
print(df["anomaly_strength"].max())


# Visualization
plt.figure(figsize=(12,6))

plt.plot(df["date"], df["smooth_strength"],label = "Smoothened Anomaly Strength")

# Highlight top anomalies
strong = df[df["anomaly_strength"] > 0.2]

plt.scatter(
    strong["date"],
    strong["anomaly_strength"],
    color="red",
    label="Strong Anomalies",
    s= 50      # Size of scatter dots

)

plt.title("Anomaly Strength over time")
plt.xlabel("Date")
plt.ylabel("Strength")
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig("Plots/Strongest_Anomaly_Detection")
plt.show()
