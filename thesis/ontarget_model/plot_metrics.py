#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metrics = pd.read_csv("all_except_hct116/ont_all_except_hct116_model_training_metrics.csv")
plt.figure(figsize=(12, 8))
plt.yticks(np.arange(0, 1.1, step=0.1))  # Set label locations.
plt.ylim(0, 1)
plt.plot(metrics["train_spearman"], label="Train Spearman")
plt.plot(metrics["val_spearman"], label="Val Spearman")
# plt.plot(metrics["test_Spearman"], label="Test Spearman")
plt.xlabel("Epochs")
plt.ylabel("Spearman")
plt.title("Spearman over Epochs")
plt.legend()
plt.savefig("all_except_hct116/all_except_hct116_Spearman_metrics.png")
