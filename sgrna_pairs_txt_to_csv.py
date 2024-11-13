#!/usr/bin/env python3

import csv

# File paths
input_file = 'sgRNA_efficacies_pairs5.txt'
output_file = 'sgRNA_efficacies_pairs5.csv'

# List to store the extracted values
numerical_values = []

# Reading the input file
with open(input_file, 'r') as file:
    for line in file:
        # Extract numerical value using string operations
        start_idx = line.find('[[')
        end_idx = line.find(']]')
        if start_idx != -1 and end_idx != -1:
            value = line[start_idx + 2:end_idx]
            numerical_values.append(value)

# Writing to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for value in numerical_values:
        writer.writerow([value])

print(f"Extraction complete! Numerical values have been saved to '{output_file}'.")
