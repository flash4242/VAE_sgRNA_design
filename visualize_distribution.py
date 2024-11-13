#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Read the CSV file into a DataFrame
df = pd.read_csv('sgRNA_efficacies_pairs5.csv', header=None)

# Rename the first column to 'Efficacy' for clarity
df.columns = ['Efficacy']

# Display the first few rows of the DataFrame to check the data
print(df.head())

# Plot the histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df['Efficacy'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Efficacy Values')
plt.xlabel('Efficacy')
plt.ylabel('Frequency')
plt.grid(True)
plt.ylim(0, 190)
plt.xlim(0, 1)
plt.xticks([0, 0.5, 1])

plt.xticks(np.arange(0, 1, step=0.1))


plt.savefig("dist_vae5.png")
plt.show()
