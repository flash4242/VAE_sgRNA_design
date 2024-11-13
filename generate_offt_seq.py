#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
from crispr_offt_prediction import crispr_offt


offt_model = crispr_offt()
offt_model.load_weights("crispr_offt.h5")
for layer in offt_model.layers:
        layer.trainable = False

# Define the nucleotide pair encoding dictionary for off-target profile prediction
pair_encoding = {
    "AA": 0, "AC": 1, "AG": 2, "AT": 3,
    "CA": 4, "CC": 5, "CG": 6, "CT": 7,
    "GA": 8, "GC": 9, "GG": 10, "GT": 11,
    "TA": 12, "TC": 13, "TG": 14, "TT": 15
}


def generate_offt_sequence(sequence):
    """Generate a similar sequence with 1-2 character changes."""
    sequence = list(sequence)  # Convert to list for mutability
    mutation_count = random.choice([1, 2])  # Randomly choose 1 or 2 mutations
    for _ in range(mutation_count):
        idx = random.randint(0, len(sequence) - 1)  # Random position in the sequence
        original_base = sequence[idx]
        new_base = random.choice([base for base in 'ACTG' if base != original_base])
        sequence[idx] = new_base  # Apply mutation
    return ''.join(sequence)

# Function to encode an target-sgRNA - OT pair
def encode_pairwise(sgrna, OT):
    encoded_sequence = []
    for s, d in zip(sgrna, OT):
        pair = s + d
        encoded_sequence.append(pair_encoding[pair])  # Convert each integer to a string
    return np.array([encoded_sequence])


def process_sequences(input_file, output_file):
    # Load sequences from CSV
    df = pd.read_csv(input_file)
    original_sequences = df['Sequence'].tolist()

    offt_results = []

    for seq in original_sequences:
        offt_seq = generate_offt_sequence(seq)
        encoded_features = encode_pairwise(seq, offt_seq)
        y_pred_offt = offt_model.predict(encoded_features)
        offt_results.append((seq, offt_seq, y_pred_offt))


    # Create a new DataFrame and save to CSV
    result_df = pd.DataFrame(offt_results, columns=['Original Sequence', 'Offt Sequence', 'Offt efficacy'])
    result_df.to_csv(output_file, index=False)

# Usage
process_sequences("sgrnas_5.csv", "sgrna_5_with_offt.csv")
