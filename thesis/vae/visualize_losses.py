#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

losses = pd.read_csv("extended_vae_losses.csv")
plt.figure(figsize=(12, 8))
#plt.yticks(np.arange(0, 1.1, step=0.1))  # Set label locations.
#plt.ylim(0, 1)
plt.plot(losses["train_vae_loss"], label="Train loss")
plt.plot(losses["val_vae_loss"], label="Val loss")
plt.plot(losses["train_ontarget_loss"], label="Train on-target loss")
plt.plot(losses["val_ontarget_loss"], label="Val on-target loss")

# plt.plot(metrics["test_Spearman"], label="Test Spearman")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and validation loss over Epochs")
plt.legend()
plt.savefig("extended_vae_losses.png")
