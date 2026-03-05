# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("Processed_Data/Yearly_temperature.csv")

# Decade Calculation
#Decade Find out
df["decade"] = (df["year"]//10)*10
print(df.head())

#Plotting Decade vs Avg temp
df_decadal_avg = df.groupby("decade").agg(mean_temp=("mean_temp","mean")).reset_index()
print(df_decadal_avg)


# Calculate baseline (Long-Term Mean Temp)
baseline_temp = df["mean_temp"].mean()
print("Baseline temperature is: ",baseline_temp)

# Calculate Temperature Anamoly
df["temp_anomaly"] = df["mean_temp"] - baseline_temp

# Plot temperature anomalies over time
plt.figure(figsize=(10,6))

plt.plot(df["year"],df["temp_anomaly"],marker = "o",linestyle ="-",color = "red", label="Temperature Anamoly")

# Need to Plot a zero reference line to visualize the deviations
plt.axhline(0, linestyle = "--", color = "black", label ="Normal")

plt.xlabel("Year")
plt.ylabel("Temperature Anamoly in Celsius")
plt.title("Yearly Temperature Anamolies")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure first
plt.savefig("Plots/Temperature_Anomalies.png")

plt.show()


