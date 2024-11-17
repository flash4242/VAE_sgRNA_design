#!/usr/bin/env python3

import csv

# Define the filename
filename = "sgRNA_efficacies_pairs5.csv"

# Initialize a counter for values greater than 0
count = 0

# Open the file and read the values
with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for value in row:
            if float(value) > 0:  # Convert the value to a float and check
                count += 1

# Print the result
print(f"Number of values greater than 0: {count}")
