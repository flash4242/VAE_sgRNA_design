#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv("ont_model_training_metrics.csv")
plt.figure(figsize=(12, 8))
plt.plot(metrics["train_mse"], label="Train MSE")
plt.plot(metrics["val_mse"], label="Val MSE")
plt.plot(metrics["test_mse"], label="Test MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.legend()
plt.savefig("metrics.png")
