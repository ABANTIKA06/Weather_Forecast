# Converting Hourly to Daily time in new_history.csv

# Import Libraries
import pandas as pd

# Load the CSV File
df1 = pd.read_csv("Data_Raw/history.csv")
df2 = pd.read_csv("Data_Raw/new_history.csv")

# Convert time columns to datetime
df1["time"] = pd.to_datetime(df1["time"])
df2["time"] = pd.to_datetime(df2["time"])

# Extract date column in df2
df2["date"] = df2["time"].dt.date

# Convert hourly -> daily
df2_daily = df2.groupby("date").mean(numeric_only=True).reset_index()

# Extract date column in df1
df1["date"] = df1["time"].dt.date

# Merge
df = pd.merge(df1, df2_daily, on="date")
print(df.head())


#Rename columns
df = df.rename(columns={
  "temperature_2m_mean (°C)": "mean_temp",
  "temperature_2m_max (°C)": "max_temp",
  "temperature_2m_min (°C)": "min_temp",
  "precipitation (mm)": "precipitation",
  "cloud_cover (%)": "cloud_cover"
})

# Temperature Range(Daily Variation)
df["temp_range"] = df["max_temp"] - df["min_temp"]

# Temperature Average Change (Day to Day change)
df["temp_average_change"] = df["mean_temp"].diff()

# Fill First NaN value with 0
df["temp_avg_change"] = df["temp_average_change"].fillna(0)

# Select Final columns
final_df = df[["date","mean_temp",
"max_temp","min_temp",
"precipitation","cloud_cover",
"temp_range","temp_avg_change"]]

# Save New Dataset
final_df.to_csv("Processed_Data/final_history.csv",index=False)

print("Final history.csv created successfully")
