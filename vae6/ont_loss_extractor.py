#!/usr/bin/env python3

import csv

# Input and output filenames
input_file = 'sgRNA_efficacies_pairs6.txt'
output_file = 'sgRNA_efficacies_pairs6.csv'

# Open the input file for reading and output file for writing
with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
    # Create a CSV writer
    writer = csv.writer(csv_file)

    # Read each line from the text file
    for line in txt_file:
        # Split the line by commas and extract the middle value
        parts = line.strip().split(',')
        if len(parts) > 2:
            middle_value = parts[1]
            # Write the middle value to the CSV file
            writer.writerow([middle_value])

print("Values have been successfully extracted and written to output.csv.")
