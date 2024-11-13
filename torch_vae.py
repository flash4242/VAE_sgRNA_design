#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Define constants
SEQ_LEN = 23  # Length of sgRNA sequences
VOCAB_SIZE = 4  # A, C, G, T

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# One-hot encoding for sgRNA sequences
def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((SEQ_LEN, VOCAB_SIZE))
    for i, char in enumerate(sequence):
        encoded[i, mapping[char]] = 1
    return encoded

def one_hot_decode(encoded_seq):
    mapping = ['A', 'C', 'G', 'T']
    decoded = ''.join([mapping[np.argmax(vec)] for vec in encoded_seq])
    return decoded

# Dataset for sgRNA sequences
class sgrnaDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.sequences = df['sgRNA'].values
        self.encoded_sequences = [one_hot_encode(seq) for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_sequences[idx], dtype=torch.float32)

# VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # To output probabilities for each nucleotide
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# Loss function
def vae_loss(reconstructed, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction='sum')(reconstructed, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training function
def train_vae(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.view(batch.size(0), -1).to(device)  # Flatten input and move to device
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset)}')

# Load data
dataset = sgrnaDataset('sgrna_hela_vae.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the VAE model
input_dim = SEQ_LEN * VOCAB_SIZE
latent_dim = 16  # Adjust based on desired latent space size
vae_model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)  # Move model to device

# Optimizer
optimizer = optim.Adam(vae_model.parameters(), lr=0.001)

# Train the VAE
train_vae(vae_model, dataloader, optimizer, device, epochs=20)

# Sampling new sgRNAs
def generate_new_sgRNA(model, num_samples=5):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)  # Move latent variable to device
        decoded_sgrnas = model.decode(z).view(num_samples, SEQ_LEN, VOCAB_SIZE)
        decoded_sgrnas = decoded_sgrnas.cpu().numpy()
        new_sgrnas = [one_hot_decode(seq) for seq in decoded_sgrnas]
        return new_sgrnas

# Generate new sgRNAs
new_sgRNAs = generate_new_sgRNA(vae_model, num_samples=5)
print("Generated sgRNAs:")
for sgRNA in new_sgRNAs:
    print(sgRNA)