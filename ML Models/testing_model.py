import matplotlib.pyplot as plt
import torch
from CNN_Extended import CNNFingerprinter
from model_training import testing_dataloader  # assuming you defined this there
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Load Model ======
model = CNNFingerprinter().to(device)
model.load_state_dict(torch.load('RF_Model_Weights.pth'))
model.eval()

# ====== Initialize Storage ======
num_classes = 16
correct_per_class = [0] * num_classes
total_per_class = [0] * num_classes

# ====== Run Validation ======
with torch.no_grad():
    for (IQ, labels) in testing_dataloader:
        IQ = IQ.to(device)
        labels = labels.to(device)

        outputs = model(IQ)
        preds = torch.argmax(outputs, dim=1)

        batch_size = labels.size(0)
        for i in range(batch_size):
            true = labels[i].item()
            pred = preds[i].item()
            total_per_class[true] += 1
            if true == pred:
                correct_per_class[true] += 1

# ====== Compute Accuracy per Transmitter ======
accuracy_per_class = [
    correct / total if total > 0 else 0.0
    for correct, total in zip(correct_per_class, total_per_class)
]

# ====== Plot Bar Chart ======
fig, ax = plt.subplots(figsize=(12, 6))
x_labels = [f"TX {i}" for i in range(num_classes)]

ax.bar(x_labels, accuracy_per_class, color='steelblue')
ax.set_ylim([0, 1.0])
ax.set_ylabel("Accuracy")
ax.grid(axis='y')

for i, acc in enumerate(accuracy_per_class):
    ax.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
