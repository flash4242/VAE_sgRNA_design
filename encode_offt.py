#!/usr/bin/env python3

import pandas as pd

# Define the nucleotide pair encoding dictionary
pair_encoding = {
    "AA": 0, "AC": 1, "AG": 2, "AT": 3,
    "CA": 4, "CC": 5, "CG": 6, "CT": 7,
    "GA": 8, "GC": 9, "GG": 10, "GT": 11,
    "TA": 12, "TC": 13, "TG": 14, "TT": 15
}

# Load the CSV file into a DataFrame
df = pd.read_csv('sgrna_5_with_offt.csv')

# Function to encode an target-sgRNA - OT pair
def encode_pairwise(sgRNA, OT):
    encoded_sequence = []
    for s, d in zip(sgRNA, OT):
        pair = s + d
        encoded_sequence.append(str(pair_encoding[pair]))  # Convert each integer to a string
    return ';'.join(encoded_sequence)  # Join the list of strings with commas

# Apply the encoding to the sgRNA-DNA pairs
df['Encoded Sequence'] = df.apply(lambda row: encode_pairwise(row['Original Sequence'], row['Mutated Sequence']), axis=1)

# Display the resulting DataFrame
print(df[['Original Sequence', 'Mutated Sequence', 'Encoded Sequence']])

# Optionally save to a new CSV file
df.to_csv('sgrna_5_offt_encoded.csv', index=False, quoting=3)
