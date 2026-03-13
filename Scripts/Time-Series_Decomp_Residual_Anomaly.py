# Import Libraries
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt 

# Seasonal Decompose Library for splitting time series into trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose 

# Read CSV File
df = pd.read_csv("Processed_Data/Yearly_temperature.csv")

# Convert Year to Time Index 
df_time_index = df.set_index("year")
df_time_index.head()

# Decompose the time series into TRend, sesonality and residual component
decomposition = seasonal_decompose(
    df_time_index["mean_temp"],
    model="additive",
    period=5
)

# Plot Decomposition
#decomposition.plot()
#plt.show()

# Extract Components(Year|mean_temp|trend|seasonality|residual)
df["trend"] = decomposition.trend.values
df["seasonality"] = decomposition.seasonal.values
df["residual"] = decomposition.resid.values


# Detect Residual Anomalies
# Calculate residual standard deviation
residual_std = df["residual"].std()
print("The residual standard deviation is: ",residual_std)

# Mark Anomalies
df["residual_anomaly"] = np.where(abs(df["residual"]) > 1.5 * residual_std,1, 0)

# Save final output to CSV
df.to_csv("Processed_Data/Time_Series_Decompose.csv",index=False)

# Drop NaN Values
df = df.dropna()

#print(df.head(20))
print(df[["year","mean_temp","trend", "seasonality", "residual","residual_anomaly"]].head(30))

# Visualize Residual Component
plt.figure(figsize=(12,6))

plt.plot(df["year"],df["residual"],label = "Residual Component",color="Blue")

# Plot Horizontal line over X-Axis
plt.axhline(0, color="black", linestyle = "--")

plt.xlabel("Year")
plt.ylabel("Residual Temperature")
plt.title("Residual Component of Temperature")
plt.legend()
plt.grid(True,alpha =1)

plt.savefig("Plots/Time-Series_Residual")
plt.show()

################################################################
# Plot Residuals with Red Anomaly Points

plt.figure(figsize=(12,6))

# Plot Residual Line
plt.plot(df["year"],df["residual"],label ="Residual Component",color ="Blue")

# Add Zero Reference Line
plt.axhline(0, linestyle ="--", color ="Black")

# Extract Anomaly Rows
anomalies = df[df["residual_anomaly"]== 1]

# Plot Anomalies
plt.scatter(
    anomalies["year"],
    anomalies["residual"],
    color = "red",
    s = 80,
    label = "Anomaly"
)

plt.xlabel("Year")
plt.ylabel("Residual Temperature")
plt.title ("Residual Temperature with Anomaly Detection")
plt.legend()
plt.grid(True)

plt.savefig("Plots/Red_Residual_Anomaly_Points")
plt.show()



