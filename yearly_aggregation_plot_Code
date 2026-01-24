import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the yearly temperature data
df = pd.read_csv("Processed/Yearly_temperature.csv")

# Create a figure with a larger size for better visibility
plt.figure(figsize=(14, 8))

# Plot all three temperature lines
plt.plot(df['year'], df['mean_temp'], marker='o', linewidth=2, label='Mean Temperature', color='#2E86AB', markersize=5)
plt.plot(df['year'], df['max_temp'], marker='s', linewidth=2, label='Max Temperature', color='#E63946', markersize=5)
plt.plot(df['year'], df['min_temp'], marker='^', linewidth=2, label='Min Temperature', color='#06A77D', markersize=5)

# Customize the plot
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
plt.title('Long-Term Warming Trend Analysis (1994-2025)', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add minor gridlines
plt.minorticks_on()
plt.grid(which='minor', alpha=0.1, linestyle=':')

# Rotate x-axis labels for better readability
plt.xticks(df['year'][::2], rotation=45, ha='right')  # Show every 2nd year

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('Processed/warming_trend_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'Processed/warming_trend_plot.png'")

# Display the plot
plt.show()


