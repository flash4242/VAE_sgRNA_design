#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import wandb
import os
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("ont1", exist_ok=True)
wandb.login(key="628545f3ecebbb741774074b3331ffdb3e4ad1fd")

# Initialize Weights & Biases
wandb.init(
    project='own-ontarget-pred',
    entity='nagydavid02-bme',
    config={
        "dropout_rate": 0.3,
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 300,
        "patience": 20,
        "train_split": 0.7,
        "validation_split": 0.15,
    }
)
config = wandb.config

# Define dataset class
class SgRNADataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.sequence_data = self.data["sgRNA"]
        self.targets = self.data["Normalized efficacy"].values.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq = self.sequence_data[idx]
        target = self.targets[idx]
        
        # Encode bases as one-hot channels
        base_map = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        encoded_seq = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in base_map:
                encoded_seq[base_map[base], i] = 1.0
        return torch.tensor(encoded_seq), torch.tensor(target)

# Load dataset and create DataLoaders
dataset = SgRNADataset('data_for_ont_training.csv')
train_size = int(config.train_split * len(dataset))
val_size = int(config.validation_split * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size)
test_loader = DataLoader(test_set, batch_size=config.batch_size)

# Define the model
class BaselineConvModel(nn.Module):
    def __init__(self):
        super(BaselineConvModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        #self.dropout = nn.Dropout(config.dropout_rate)
        
        # Assuming 23 as original length -> 23 -> 11 (First pooling reduces the length from 23 to 11 (since 23//2=11 - rounds down)) -> 5 (=11//2=5 after second pool)
        self.fc1 = nn.Linear(16 * 5, 32)  # Adjusted based on pooling
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Apply conv1, batch norm, relu, and max pool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Apply conv2, batch norm, relu, and max pool
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x)) # x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)



# Initialize model, loss function, and optimizer
model = BaselineConvModel().to(device)
summary(model, input_size=(4, 23), device=str(device))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Initialize variables for early stopping and best model checkpoint
best_val_loss = float('inf')
best_model_path = "ont1/best_model.pth"

# Training and validation loop
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * sequences.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item() * sequences.size(0)
    val_loss /= len(val_loader.dataset)

    # Log metrics to WandB
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    print(f"Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model if the validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved.")

# Final testing and logging
model.load_state_dict(torch.load(best_model_path))  # Load the best model before testing
model.eval()
test_loss = 0.0
with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item() * sequences.size(0)
test_loss /= len(test_loader.dataset)

wandb.log({"test_loss": test_loss})
print(f"Test Loss: {test_loss:.4f}")

wandb.finish()

# Predicting efficacy on new sgRNAs
def load_model(model_path="ont1/best_model.pth"):
    model = BaselineConvModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_sgRNA(model, sgRNA_seq):
    # Encode the sgRNA sequence
    base_map = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    encoded_seq = np.zeros((4, len(sgRNA_seq)), dtype=np.float32)
    for i, base in enumerate(sgRNA_seq):
        if base in base_map:
            encoded_seq[base_map[base], i] = 1.0
    input_tensor = torch.tensor(encoded_seq).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)  # Move input to the correct device

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

# Example usage:
model = load_model()
new_sgRNA = "CTTGCTCGCGCAGGACGAGGCGG"  # Replace with actual sgRNA sequence
predicted_efficacy = predict_sgRNA(model, new_sgRNA)
print(f"Predicted Normalized Efficacy: {predicted_efficacy}")

