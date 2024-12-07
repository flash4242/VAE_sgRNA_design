#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import pandas as pd
from model import BaselineConvModel  # Import the model class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Weights & Biases
wandb.login(key="628545f3ecebbb741774074b3331ffdb3e4ad1fd")
wandb.init(
    project="sgRNA-extended-vae",
    entity="nagydavid02-bme",
    config={
        "latent_dim": 32,
        "embedding_dim": 32,
        "lstm_units": [128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
        "validation_split": 0.2,
        "ontarget_loss_weight": 1000,
        "generated_sgrnas": 100
    },
)
config = wandb.config

# -----------------  ON-TARGET MODEL HELPER FUNCTIONS --------------------
# Modify the VAE Loss Function
def on_target_loss(predicted_efficacy):
    ground_truth = 1.0  # Assuming the ground truth efficacy is 1
    return (predicted_efficacy - ground_truth) ** 2

def load_ont_model(model_path="best_ont_model.pth"):
    model = BaselineConvModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_sgRNA_eff(model, sgRNA_seq):
    base_map = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    encoded_seq = np.zeros((4, len(sgRNA_seq)), dtype=np.float32)
    for i, base in enumerate(sgRNA_seq):
        if base in base_map:
            encoded_seq[base_map[base], i] = 1.0
    input_tensor = torch.tensor(encoded_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()
# -----------------  ON-TARGET MODEL HELPER FUNCTIONS - END --------------------

# Define dataset for sgRNAs
class sgRNADataset(Dataset):
    def __init__(self, sgRNAs, char_to_int, max_length):
        self.sgRNAs = sgRNAs
        self.char_to_int = char_to_int
        self.max_length = max_length
        self.encoded_sgRNAs = [
            [char_to_int[char] for char in sgRNA] for sgRNA in sgRNAs
        ]
        self.padded_sgRNAs = [
            seq + [0] * (max_length - len(seq)) for seq in self.encoded_sgRNAs
        ]

    def __len__(self):
        return len(self.padded_sgRNAs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.padded_sgRNAs[idx], dtype=torch.long)
        return seq, seq  # Input and target are the same for VAE


# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim, vocab_size, embedding_dim, lstm_units, dropout_rate, max_length):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm1 = nn.LSTM(embedding_dim, lstm_units[0], batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units[0], lstm_units[1], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.z_mean = nn.Linear(lstm_units[1], latent_dim)
        self.z_log_var = nn.Linear(lstm_units[1], latent_dim)

        self.decoder_dense = nn.Linear(latent_dim, lstm_units[1])
        self.decoder_lstm1 = nn.LSTM(lstm_units[1], lstm_units[0], batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units[0], embedding_dim, batch_first=True)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.max_length = max_length

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + epsilon * torch.exp(0.5 * z_log_var)

    def encode(self, x):
        x = self.embedding(x)
        x, _ = self.encoder_lstm1(x)
        x = self.dropout(x)
        x, _ = self.encoder_lstm2(x)
        x = self.dropout(x)
        z_mean = self.z_mean(x[:, -1])
        z_log_var = self.z_log_var(x[:, -1])
        return z_mean, z_log_var, self.reparameterize(z_mean, z_log_var)

    def decode(self, z):
        x = self.decoder_dense(z).unsqueeze(1).repeat(1, self.max_length, 1)
        x, _ = self.decoder_lstm1(x)
        x = self.dropout(x)
        x, _ = self.decoder_lstm2(x)
        x = self.output_layer(x)
        return x

    def forward(self, x):
        z_mean, z_log_var, z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var


# # Loss Function
# def vae_loss(reconstructed, target, z_mean, z_log_var):
#     reconstruction_loss = nn.CrossEntropyLoss(reduction="sum")(reconstructed.permute(0, 2, 1), target)
#     kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
#     kl_loss /= inputs.size(0)  # Normalize KL loss by batch size
#     return reconstruction_loss + kl_loss


# Load and preprocess data
data = pd.read_csv("all_data_vae_training.csv")
sgRNAs = data["sgRNA"].values
nucleotides = ["A", "C", "G", "T"]
char_to_int = {nucleotide: i for i, nucleotide in enumerate(nucleotides)}
int_to_char = {i: nucleotide for i, nucleotide in enumerate(nucleotides)}
max_length = max(len(sgRNA) for sgRNA in sgRNAs)

dataset = sgRNADataset(sgRNAs, char_to_int, max_length)
train_size = int(len(dataset) * (1 - config.validation_split))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize model, optimizer, and device
model = VAE(
    latent_dim=config.latent_dim,
    vocab_size=len(nucleotides),
    embedding_dim=config.embedding_dim,
    lstm_units=config.lstm_units,
    dropout_rate=config.dropout_rate,
    max_length=max_length,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Training Loop with sgRNA Generation and Evaluation
best_val_loss = float("inf")

# Load the frozen on-target model
ontarget_model = load_ont_model()  # Ensure this is the pre-trained on-target prediction model
ontarget_model.eval()

# Training Loop
# Initialize variables to log losses
train_vae_losses, train_ontarget_losses, train_total_losses = [], [], []
val_vae_losses, val_ontarget_losses, val_total_losses = [], [], []

# ----------------- Modified Loss Function --------------------
# Modify the VAE Loss Function to include on-target efficacy loss
def vae_loss_with_ontarget(reconstructed, target, z_mean, z_log_var, reconstructed_sgRNAs, ontarget_model):
    # Reconstruction loss
    reconstruction_loss = nn.CrossEntropyLoss(reduction="sum")(reconstructed.permute(0, 2, 1), target)

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    kl_loss /= target.size(0)  # Normalize KL loss by batch size

    # On-target efficacy loss
    total_ontarget_loss = 0.0
    for sgRNA in reconstructed_sgRNAs:
        predicted_efficacy = predict_sgRNA_eff(ontarget_model, sgRNA)
        total_ontarget_loss += on_target_loss(predicted_efficacy)

    avg_ontarget_loss = total_ontarget_loss / len(reconstructed_sgRNAs)
    avg_ontarget_loss *= config.ontarget_loss_weight  # Scale the loss

    # Total loss
    total_loss = reconstruction_loss + kl_loss + avg_ontarget_loss
    vae_loss = reconstruction_loss + kl_loss
    return total_loss, vae_loss, avg_ontarget_loss


# ----------------- Training Loop with Reconstructed sgRNAs --------------------
for epoch in range(config.epochs):
    model.train()
    train_ontarget_loss = 0.0
    train_total_loss = 0.0
    train_vae_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass through the VAE
        reconstructed, z_mean, z_log_var = model(inputs)

        # Decode reconstructed sgRNAs to sequences
        reconstructed_indices = torch.argmax(reconstructed, dim=-1)
        reconstructed_sgRNAs = [
            "".join(int_to_char[idx.item()] for idx in sequence)
            for sequence in reconstructed_indices
        ]

        # Compute total loss with on-target efficacy
        total_loss, vae_loss, ont_loss = vae_loss_with_ontarget(
            reconstructed, targets, z_mean, z_log_var, reconstructed_sgRNAs, ontarget_model
        )

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Track losses
        train_vae_loss += vae_loss
        train_ontarget_loss += ont_loss
        train_total_loss += total_loss.item()

    # Normalize losses by dataset size
    train_vae_loss /= len(train_loader.dataset)
    train_ontarget_loss /= len(train_loader.dataset)
    train_total_loss /= len(train_loader.dataset)

    # Validation Step
    model.eval()
    val_vae_loss= 0.0
    val_ontarget_loss = 0.0
    val_total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            reconstructed, z_mean, z_log_var = model(inputs)

            # Decode reconstructed sgRNAs to sequences
            reconstructed_indices = torch.argmax(reconstructed, dim=-1)
            reconstructed_sgRNAs = [
                "".join(int_to_char[idx.item()] for idx in sequence)
                for sequence in reconstructed_indices
            ]

            # Compute total loss with on-target efficacy
            total_loss, vae_loss, ont_loss = vae_loss_with_ontarget(
                reconstructed, targets, z_mean, z_log_var, reconstructed_sgRNAs, ontarget_model
            )

            # Track losses
            val_vae_loss += vae_loss
            val_ontarget_loss += ont_loss
            val_total_loss += total_loss.item()

    # Normalize losses by dataset size
    val_vae_loss /= len(val_loader.dataset)
    val_ontarget_loss /= len(val_loader.dataset)
    val_total_loss /= len(val_loader.dataset)

    # Log losses to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_vae_loss": train_vae_loss,
        "train_ontarget_loss": train_ontarget_loss,
        "train_total_loss": train_total_loss,
        "val_vae_loss": val_vae_loss,
        "val_ontarget_loss": val_ontarget_loss,
        "val_total_loss": val_total_loss,
    })

    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        torch.save(model.state_dict(), "best_extended_vae.pth")
        wandb.run.summary["best_val_loss"] = best_val_loss

# Save all losses to a CSV file
losses_df = pd.DataFrame({
    "epoch": list(range(1, config.epochs + 1)),
    "train_vae_loss": train_vae_losses,
    "train_ontarget_loss": train_ontarget_losses,
    "train_total_loss": train_total_losses,
    "val_vae_loss": val_vae_losses,
    "val_ontarget_loss": val_ontarget_losses,
    "val_total_loss": val_total_losses,
})
losses_df.to_csv("extended_vae_losses.csv", index=False)
#print("All losses saved to 'extended_vae_losses.csv'.")

# Load the best model
model.load_state_dict(torch.load("best_extended_vae.pth"))
model.eval()

# Testing: Generate 1000 sgRNAs and save on-target efficacies
generated_sgRNAs = []
predicted_efficacies = []

with torch.no_grad():
    for _ in range(200):  # 200 batches of size 5 to generate 1000 sgRNAs
        z = torch.randn(5, config.latent_dim).to(device)
        generated_outputs = model.decode(z)
        generated_indices = torch.argmax(generated_outputs, dim=-1)

        # Convert generated indices to sgRNA sequences
        for sequence in generated_indices:
            sgRNA = "".join(int_to_char[idx.item()] for idx in sequence)
            generated_sgRNAs.append(sgRNA)
            efficacy = predict_sgRNA_eff(ontarget_model, sgRNA)
            predicted_efficacies.append(efficacy)

# Save results to a CSV file
import pandas as pd

results_df = pd.DataFrame(predicted_efficacies)
results_df.to_csv("extended_vae_results.csv", index=False)

#print("Generated 1000 sgRNAs and saved results to 'extended_vae_results.csv'.")
