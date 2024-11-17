#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Weights & Biases
wandb.login(key="628545f3ecebbb741774074b3331ffdb3e4ad1fd")
wandb.init(
    project="sgRNA-vanilla-vae",
    entity="nagydavid02-bme",
    config={
        "latent_dim": 32,
        "embedding_dim": 32,
        "lstm_units": [128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 2000,
        "validation_split": 0.2,
    },
)
config = wandb.config

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


# Loss Function
def vae_loss(reconstructed, target, z_mean, z_log_var):
    reconstruction_loss = nn.CrossEntropyLoss(reduction="sum")(reconstructed.permute(0, 2, 1), target)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    kl_loss /= inputs.size(0)  # Normalize KL loss by batch size
    return reconstruction_loss + kl_loss


# Load and preprocess data
data = pd.read_csv("sgrna_hela_vae.csv")
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

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training Loop
best_val_loss = float("inf")
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        reconstructed, z_mean, z_log_var = model(inputs)
        loss = vae_loss(reconstructed, targets, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            reconstructed, z_mean, z_log_var = model(inputs)
            loss = vae_loss(reconstructed, targets, z_mean, z_log_var)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_vae.pth")
        wandb.run.summary["best_val_loss"] = best_val_loss

    print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load the best model
model.load_state_dict(torch.load("best_vae.pth"))
model.eval()

wandb.finish()
