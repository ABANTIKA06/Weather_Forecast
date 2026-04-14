# Import Libraries
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV File
df = pd.read_csv("Processed_Data/Time_Series_Decompose.csv")

# Pick the mean_temp column in rowsxcol or Table Format 
x = df[["mean_temp"]]
print(x)

# Create Model
model = IsolationForest(contamination=0.1, random_state =1)

# Detect Anomalies
df["isolation_anomaly"] = model.fit_predict(x)

# Convert Result
df["isolation_anomaly"] = np.where(df["isolation_anomaly"]== -1, 1, 0)
print(df[["year", "mean_temp", "isolation_anomaly"]])

# Visualization
plt.figure(figsize=(12,6))

plt.plot(df["year"],df["mean_temp"],label ="Mean Temperature",color ="blue")

# Extract Anomaly rows
anomalies = df[df["isolation_anomaly"]==1]

# Plot Anomalies
plt.scatter(
    anomalies["year"],
    anomalies["mean_temp"],
    color= "red",
    s=80,
    label= "Isolation Forest Anomaly"    
)

plt.xlabel("Year"),
plt.ylabel("Mean_Temperature"),
plt.title("Isolation Forest Temperature Anomaly Detection")
plt.legend()
plt.grid(True)

plt.savefig("Plots/Isolation_Forest_Anomaly")
plt.show()
#################################################

#Combined Residual and Isolation Anomaly together Plot
plt.figure(figsize=(12,6))

# Plot temperature line
plt.plot(df["year"], df["mean_temp"],
         label="Mean Temperature",
         color="blue")

# Residual anomalies
residual_anoms = df[df["residual_anomaly"] == 1]

plt.scatter(
    residual_anoms["year"],
    residual_anoms["mean_temp"],
    color="orange",
    s=80,
    label="Residual Anomaly"
)

# Isolation Forest anomalies
iso_anoms = df[df["isolation_anomaly"] == 1]

plt.scatter(
    iso_anoms["year"],
    iso_anoms["mean_temp"],
    color="red",
    s=100,
    marker="x",
    label="Isolation Forest Anomaly"
)

plt.xlabel("Year")
plt.ylabel("Mean Temperature (°C)")
plt.title("Comparison of Anomaly Detection Methods")

plt.legend()
plt.grid(True)

plt.savefig("Plots/Combined_Anomaly_Comparison")
plt.show()