#!/usr/bin/env python3
import torch
import numpy as np
from model import BaselineConvModel  # Import the model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="best_ont_model.pth"):
    model = BaselineConvModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_sgRNA(model, sgRNA_seq):
    base_map = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    encoded_seq = np.zeros((4, len(sgRNA_seq)), dtype=np.float32)
    for i, base in enumerate(sgRNA_seq):
        if base in base_map:
            encoded_seq[base_map[base], i] = 1.0
    input_tensor = torch.tensor(encoded_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

if __name__ == "__main__":
    model = load_model()
    new_sgRNA = "CTTGCTCGCGCAGGACGAGGCGG"
    predicted_efficacy = predict_sgRNA(model, new_sgRNA)
    print(f"Predicted Normalized Efficacy: {predicted_efficacy}")
