#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file
df = pd.read_csv("Processed/Yearly_temperature.csv")

#Show the list of all the columns
print(df.columns.tolist())

#Decade Find out
df["decade"] = (df["year"]//10)*10
print(df.head())

#Plotting Decade vs Avg temp
df_decadal_avg = df.groupby("decade").agg(mean_temp=("mean_temp","mean")).reset_index()
print(df_decadal_avg)

#Plotting decade VS Mean Temp
plt.figure(figsize=(10,6))
plt.plot(df_decadal_avg["decade"],df_decadal_avg["mean_temp"],marker="o",linestyle="-",color="b")
plt.title("Decadal Average Temperature")
plt.xlabel("Decade")
plt.ylabel("Mean Temperature")
plt.grid(True)

#SAve the plot
plt.savefig("Plots/Decadal_avg_temp.png")

#Show Plot
plt.show()


