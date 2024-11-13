#!/usr/bin/env python3

import re
import pandas as pd

def extract_sequences(input_file, output_file):
    sequences = []

    # Read the file and extract sequences only
    with open(input_file, 'r') as file:
        for line in file:
            # Use regex to capture only the sequence part
            match = re.match(r"([A-Z]+):", line.strip())
            if match:
                sequence = match.group(1)
                sequences.append(sequence)

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame({'Sequence': sequences})
    df.to_csv(output_file, index=False)

# Usage
extract_sequences("sgRNA_efficacies_pairs5.txt", "sgrnas_5.csv")
