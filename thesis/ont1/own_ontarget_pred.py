#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config():
    def __init__(self):
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 500
        self.train_split = 0.7
        self.validation_split = 0.15

config = Config()

class SgRNADataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.efficacies = self.data["Normalized efficacy"].values.astype(np.float32)
        
        # Preprocess all sequences once in __init__
        self.sequence_data = self.data["sgRNA"]
        self.encoded_sequences = self._encode_sequences(self.sequence_data)

    def _encode_sequences(self, sequences):
        base_map = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        encoded_seqs = []
        for seq in sequences:
            encoded_seq = np.zeros((4, len(seq)), dtype=np.float32)
            for i, base in enumerate(seq):
                if base in base_map:
                    encoded_seq[base_map[base], i] = 1.0
            encoded_seqs.append(encoded_seq)
        return np.array(encoded_seqs)

    def __len__(self):
        return len(self.efficacies)

    def __getitem__(self, idx):
        encoded_seq = self.encoded_sequences[idx]
        target = self.efficacies[idx]
        return torch.tensor(encoded_seq), torch.tensor(target)

# -------------------------- Load dataset -------------------------------
dataset = SgRNADataset('data_for_ont_training.csv')
train_size = int(config.train_split * len(dataset))
val_size = int(config.validation_split * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size)
test_loader = DataLoader(test_set, batch_size=config.batch_size)
# -------------------------- Load dataset - END -------------------------------



# ---------------------------- MODEL ------------------------
class BaselineConvModel(nn.Module):
    def __init__(self):
        super(BaselineConvModel, self).__init__()
        # Initial 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride to minimize shrinking
        self.dropout1 = nn.Dropout(config.dropout_rate)

        # Residual Block 1
        self.conv2a = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.shortcut1 = nn.Conv1d(32, 64, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 2
        self.conv3a = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(128)
        self.conv3b = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(128)
        # No pooling here to preserve sequence length
        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.shortcut2 = nn.Conv1d(64, 128, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 3
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride
        self.dropout4 = nn.Dropout(config.dropout_rate)
        self.shortcut3 = nn.Conv1d(128, 256, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 4 (Additional)
        self.conv5a = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm1d(512)
        self.conv5b = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm1d(512)
        # Optional pooling here; consider removing if it shrinks too much
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout5 = nn.Dropout(config.dropout_rate)
        self.shortcut4 = nn.Conv1d(256, 512, kernel_size=1)  # To match dimensions for residual addition

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1),  # Attention weights
            nn.Softmax(dim=-1)
        )

        # Fully Connected Layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial Convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Residual Block 1
        identity1 = self.shortcut1(x)
        out = F.relu(self.bn2a(self.conv2a(x)))
        out = self.bn2b(self.conv2b(out))
        out += identity1  # Residual connection
        out = F.relu(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        # Residual Block 2
        identity2 = self.shortcut2(out)
        out = F.relu(self.bn3a(self.conv3a(out)))
        out = self.bn3b(self.conv3b(out))
        out += identity2  # Residual connection
        out = F.relu(out)
        # No pooling here to preserve sequence length
        out = self.dropout3(out)

        # Residual Block 3
        identity3 = self.shortcut3(out)
        out = F.relu(self.bn4a(self.conv4a(out)))
        out = self.bn4b(self.conv4b(out))
        out += identity3  # Residual connection
        out = F.relu(out)
        out = self.pool3(out)  # Optional pooling with reduced stride
        out = self.dropout4(out)

        # Residual Block 4 (Additional)
        identity4 = self.shortcut4(out)
        out = F.relu(self.bn5a(self.conv5a(out)))
        out = self.bn5b(self.conv5b(out))
        out += identity4  # Residual connection
        out = F.relu(out)
        out = self.pool4(out)  # Optional pooling with reduced stride
        out = self.dropout5(out)

        # Apply Attention
        weights = self.attention(out)
        out = out * weights  # Element-wise multiplication with attention weights

        # Global Average Pooling and Fully Connected Layers
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout6(out)
        out = self.fc2(out)
        return out
# ---------------------------- MODEL -END ------------------------


# Initialize model, loss function, and optimizer
model = BaselineConvModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Initialize variables for early stopping and best model checkpoint
best_val_loss = float('inf')
best_model_path = "best_ont_model.pth"

# ------------------------ Training and validation loop ------------------------
metrics = {
    "train_mse": [], "train_mae": [], "train_spearman": [], "train_pearson": [],
    "val_mse": [], "val_mae": [], "val_spearman": [], "val_pearson": [],
    "test_mse": [], "test_mae": [], "test_spearman": [], "test_pearson": []
}


for epoch in range(config.epochs):
    model.train()
    train_losses, train_maes, train_spearmans, train_pearsons = [], [], [], []
    # Training phase
    for sequences, efficacies in train_loader:
        sequences, efficacies = sequences.to(device), efficacies.to(device)
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, efficacies)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Metrics calculation
        outputs_np = outputs.cpu().detach().numpy()
        efficacies_np = efficacies.cpu().detach().numpy()
        train_maes.append(mean_absolute_error(efficacies_np, outputs_np))
        train_spearmans.append(spearmanr(efficacies_np, outputs_np).correlation)
        train_pearsons.append(pearsonr(efficacies_np, outputs_np)[0])

    # Calculate mean metrics for the training phase
    metrics["train_mse"].append(np.mean(train_losses))
    metrics["train_mae"].append(np.mean(train_maes))
    metrics["train_spearman"].append(np.mean(train_spearmans))
    metrics["train_pearson"].append(np.mean(train_pearsons))

    # Validation phase
    model.eval()
    val_losses, val_maes, val_spearmans, val_pearsons = [], [], [], []
    with torch.no_grad():
        for sequences, efficacies in val_loader:
            sequences, efficacies = sequences.to(device), efficacies.to(device)
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, efficacies)
            val_losses.append(loss.item())

            outputs_np = outputs.cpu().numpy()
            efficacies_np = efficacies.cpu().numpy()
            val_maes.append(mean_absolute_error(efficacies_np, outputs_np))
            val_spearmans.append(spearmanr(efficacies_np, outputs_np).correlation)
            val_pearsons.append(pearsonr(efficacies_np, outputs_np)[0])

    metrics["val_mse"].append(np.mean(val_losses))
    metrics["val_mae"].append(np.mean(val_maes))
    metrics["val_spearman"].append(np.mean(val_spearmans))
    metrics["val_pearson"].append(np.mean(val_pearsons))

    # Print epoch results
    print(f"Epoch [{epoch + 1}/{config.epochs}]")
    print(f"Train: MSE={metrics['train_mse'][-1]:.4f}, MAE={metrics['train_mae'][-1]:.4f}, "
          f"Spearman={metrics['train_spearman'][-1]:.4f}, Pearson={metrics['train_pearson'][-1]:.4f}")
    print(f"Val: MSE={metrics['val_mse'][-1]:.4f}, MAE={metrics['val_mae'][-1]:.4f}, "
          f"Spearman={metrics['val_spearman'][-1]:.4f}, Pearson={metrics['val_pearson'][-1]:.4f}")

    val_loss = np.mean(val_losses)
    # Save the model if the validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Model val_loss improved. Best model saved.")

    # Log test metrics in each epoch
    test_losses, test_maes, test_spearmans, test_pearsons = [], [], [], []
    with torch.no_grad():
        for sequences, efficacies in test_loader:
            sequences, efficacies = sequences.to(device), efficacies.to(device)
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, efficacies)
            test_losses.append(loss.item())

            outputs_np = outputs.cpu().numpy()
            efficacies_np = efficacies.cpu().numpy()
            test_maes.append(mean_absolute_error(efficacies_np, outputs_np))
            test_spearmans.append(spearmanr(efficacies_np, outputs_np).correlation)
            test_pearsons.append(pearsonr(efficacies_np, outputs_np)[0])

    metrics["test_mse"].append(np.mean(test_losses))
    metrics["test_mae"].append(np.mean(test_maes))
    metrics["test_spearman"].append(np.mean(test_spearmans))
    metrics["test_pearson"].append(np.mean(test_pearsons))


# ------------------------ Training and validation loop - END ------------------------


# --------------------- Measure best model's performance on test set for regplot -------------------
true_efficacies = [] # for regplot
predicted_efficacies = [] # for regplot
model.load_state_dict(torch.load(best_model_path))  # Load the best model before testing
model.eval()
with torch.no_grad():
    for sequences, efficacies in test_loader:
        sequences, efficacies = sequences.to(device), efficacies.to(device)
        outputs = model(sequences).squeeze()  # Get model predictions
        true_efficacies.append(efficacies.cpu().numpy())
        predicted_efficacies.append(outputs.cpu().numpy())
# ------------------------- Performance measure - END ---------------------------------------

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("ont_model_training_metrics.csv", index=False)
print("Metrics saved to ont_model_training_metrics.csv")

# ----------------------- REGPLOT --------------------------------
# Convert to Pandas DataFrame for Plotting
true_efficacies = np.concatenate(true_efficacies).flatten()
predicted_efficacies = np.concatenate(predicted_efficacies).flatten()
results_df = pd.DataFrame({
    "True Efficacy": true_efficacies,
    "Predicted Efficacy": predicted_efficacies
})

# Regression Plot with Seaborn
plt.figure(figsize=(8, 6))
sns.regplot(
    data=results_df, 
    x="True Efficacy", 
    y="Predicted Efficacy", 
    line_kws={"color": "red"}
)
plt.title("True vs Predicted On-Target Efficacy (Test Set)")
plt.xlabel("True Efficacy")
plt.ylabel("Predicted Efficacy")
plt.grid(True)
plt.savefig("regplot.png")
# ----------------------- REGPLOT - END --------------------------------

# ------------------------ EXAMPLE USAGE -----------------------
# Predicting efficacy on new sgRNAs
# Loading model
def load_model(model_path="best_ont_model.pth"):
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

model = load_model()
new_sgRNA = "CTTGCTCGCGCAGGACGAGGCGG"  # Replace with actual sgRNA sequence
predicted_efficacy = predict_sgRNA(model, new_sgRNA)
print(f"Predicted Normalized Efficacy: {predicted_efficacy}")
# ------------------------ EXAMPLE USAGE - END -----------------------


