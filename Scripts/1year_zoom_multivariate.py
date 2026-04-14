# Import Libraries
from sklearn.ensemble import IsolationForest
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 

# To set maximum width for features while showing the output on terminal
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
model = IsolationForest(contamination=0.01, random_state=1)

# Predict anomalies
df["multi_anomaly"] = model.fit_predict(features)

# Convert output (-1 → 1 anomaly, 0 normal)
df["multi_anomaly"] = np.where(df["multi_anomaly"] == -1, 1, 0)
print(df.head())
print(len(df))

# Zoom 1 year closely 
# Cnvert date to datetime because it is in string previously 
df["date"] = pd.to_datetime(df["date"])
print(df["date"].dtype)


# Sort Data if not sorted yet
df = df.sort_values("date")

# Zooming 1 year closely ex->2005
year =2005
df_zoom = df[df["date"].dt.year == 2005]

plt.figure(figsize=(12,6))

plt.plot(df_zoom["date"],df_zoom["mean_temp"], label ="Temperature",linewidth = 2)

anomalies = df_zoom[df_zoom["multi_anomaly"] == 1]

plt.scatter(anomalies["date"],
            anomalies["mean_temp"],
            color="red",
            s = 50,
            label="Anomalies")

plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.title("Anomaly Detection zoomed for 1 specific Year")
plt.legend()
plt.grid(True)

plt.savefig("Plots/Zoomed_year_Multivariate")
plt.show()

