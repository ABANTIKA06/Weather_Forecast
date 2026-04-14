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
model = IsolationForest(contamination=0.01, random_state=1)

# Predict anomalies
df["multi_anomaly"] = model.fit_predict(features)

# Convert output (-1 → 1 anomaly, 0 normal)
df["multi_anomaly"] = np.where(df["multi_anomaly"] == -1, 1, 0)
print(df.head())

# Smooth the plot with rolling mean and highlight anomalies 
# Cnvert date to datetime because it is in string previously
df["date"] = pd.to_datetime(df["date"])
print(df["date"].dtype)

# Sort Data if not sorted yet
df = df.sort_values("date")

# Rolling average fora smoother line in the plot
df["smooth_temp"] = df["mean_temp"].rolling(window =30).mean()

# Mention the figure SIze
plt.figure(figsize=(12,6))

# Smooth TEmperature line plot
plt.plot(df["date"],df["smooth_temp"],label="Smoothed Temperature",linewidth =2)

# HIghlight Anomalies
anomalies = df[df["multi_anomaly"]== 1]

# SCAtter plot for depicting the anomalies
plt.scatter(anomalies["date"],
            anomalies["smooth_temp"],
             color = "red",
            s= 50,
            label = "Anomalies")

plt.title("Multivariate Temperature Anomaly Detection(Smoothed)")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.grid(True)

#Save Figure
plt.savefig("Plots/Smooth_Multivariate_plot.png")
#Show figure
plt.show()

