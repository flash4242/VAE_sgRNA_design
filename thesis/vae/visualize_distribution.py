#!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# # Read the CSV file into a DataFrame
# df = pd.read_csv('vanilla_vae_results.csv', header=None)

# # Rename the first column to 'Efficacy' for clarity
# df.columns = ['Efficacy']

# # Display the first few rows of the DataFrame to check the data
# print(df.head())

# # Plot the histogram using matplotlib
# plt.figure(figsize=(10, 6))
# plt.hist(df['Efficacy'], bins=30, color='blue', edgecolor='black', alpha=0.7)
# plt.title('Distribution of Efficacy Values')
# plt.xlabel('Efficacy')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.xlim(0, 1)
# plt.ylim(0, 160)

# plt.xticks(np.arange(0, 1, step=0.1))


# plt.savefig("vanilla_vae_results.png")
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the two CSV files into DataFrames
df1 = pd.read_csv('extended_vae_results_100000.csv', header=None)
df2 = pd.read_csv('extended_vae_results.csv', header=None)

# Rename the first column to 'Efficacy' for clarity
df1.columns = ['Efficacy']
df2.columns = ['Efficacy']

# Plot the histograms using matplotlib
plt.figure(figsize=(10, 6))

# Plot the histogram for the first dataset
plt.hist(
    df1['Efficacy'], bins=30, color='orange', edgecolor='black', alpha=0.5, label='Vanilla VAE'
)

# Plot the histogram for the second dataset
plt.hist(
    df2['Efficacy'], bins=30, color='blue', edgecolor='black', alpha=0.5, label='Extended VAE'
)

# Add titles and labels
plt.title('Comparison of Efficacy Value Distributions')
plt.xlabel('Efficacy')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, max(plt.ylim()))  # Adjust the y-limit dynamically based on data
plt.xticks(np.arange(0, 1.1, step=0.1))

# Add a legend
plt.legend()

# Save and show the plot
plt.savefig("vae_comparison_results.png")
plt.show()

