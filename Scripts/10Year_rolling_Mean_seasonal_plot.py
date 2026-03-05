#Import libraries
import pandas as pd
import matplotlib.pyplot as plt

#Load the CSV file 
df = pd.read_csv("Processed/Year_Season_Mean_Temp.csv")

#Plotting the figure
plt.figure(figsize=(10,6))

#Defining Colors
colors = {
    "Winter": "lightblue",
    "Spring": "lightgreen",
    "Summer": "yellow",
    "Monsoon": "violet",
    "Autumn": "orange"
}

#Looping through all seasons in the dataframe
for season in df["season"].unique():

  #Filters Data for one season at a time
  season_df = df[df["season"] == season]

  #Arrange Data from oldest year to newest year
  season_df = season_df.sort_values("year")

  #Take Mean Temp for 10 years straight 
  season_df["rolling_10yr_mean"] = season_df["mean_temp"].rolling(window=10).mean()

  plt.plot(
    season_df["year"],
    season_df["rolling_10yr_mean"],
    marker ="o",
    label = season,
    color = colors[season]
  )

#Plotting
plt.xlabel("Year")
plt.ylabel("Seasonal 10 Year Rolling Mean Temperature")
plt.title("Seasonal 10 Year Rolling Mean Temperature Trends (Kolkata, 1994–2025)")
plt.legend()
plt.grid(True)
plt.tight_layout()


#Save plot for 10 yr rolling avg mean temp 
plt.savefig("Plots/10Year_Rolling_Mean_Temp.png")

#Show the plot
plt.show()
