import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

base_path = "checkpoint/cifar100_asym_0.9/"
files = {
    "test_acc": "cifar100_0.9_asym_acc.txt",
    "train_loss": "train_loss.txt",
    "test_loss": "test_loss.txt"
}

data = {}
for key, filename in files.items():
    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            data[key] = [float(line.strip()) for line in lines]
    else:
        data[key] = []

max_len = max(len(data[key]) for key in data)
# Pad all lists to the same length
for key in data:
    if len(data[key]) < max_len:
        data[key].extend([None] * (max_len - len(data[key])))

df = pd.DataFrame({
    "epoch": list(range(1, max_len + 1)),
    "test_accuracy": data["test_acc"],
    "train_loss": data["train_loss"],
    "test_loss": data["test_loss"],
})


df.to_csv("unicon_metrics.csv", index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["test_accuracy"], label="Test Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy over Epochs")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
plt.plot(df["epoch"], df["test_loss"], label="Test Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("unicon_dashboard.png")
print("✅ Dashboard generated: unicon_dashboard.png + unicon_metrics.csv")
