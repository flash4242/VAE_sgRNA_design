#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of CSV file names
csv_files = [
    'ont_model_training_metrics.csv',
    'ont_hct116/ont_hct116_model_training_metrics.csv',
    'ont_hek293t/ont_hek293t_model_training_metrics.csv',
    'ont_hela/ont_hela_model_training_metrics.csv',
    'ont_hl60/ont_hl60_model_training_metrics.csv'
]

# List to store test_spearman values
test_spearman_values = []

# Extract the maximum test_spearman value from each CSV
for file in csv_files:
    df = pd.read_csv(file)
    test_spearman_values.append(df['test_spearman'].max())

# Labels for the datasets
labels = ['ALL', 'HCT116', 'HEK293T', 'HELA', 'HL60']

# Adjust the positions for the bars to reduce spacing
x_positions = np.arange(len(labels))

# Create the bar plot
plt.figure(figsize=(8, 6))
plt.yticks(np.arange(0, 1.1, step=0.1))  # Set label locations.

# Plot bars with reduced spacing
plt.bar(x_positions, test_spearman_values, color='royalblue', edgecolor='black', width=0.3)

# Set x-axis ticks and labels
plt.xticks(x_positions, labels)

# Add labels and title
plt.ylabel('Spearman Correlation', fontsize=12)
plt.xlabel('Datasets', fontsize=12)
plt.title('Spearman Correlation in Various Testing Datasets', fontsize=14)
plt.ylim(0, max(test_spearman_values) * 1.1)  # Adjust y-axis for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig("spearman_comparison.png", dpi=300)
plt.show()
