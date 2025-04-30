# Purpose: Implement the training and evaluation loop for the Transformer Classifier.
# This involves loading data, initializing the model, defining loss and optimizer,
# iterating through epochs and batches, performing forward/backward passes,
# and calculating/logging metrics.

# Pseudocode:
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
# --- Your Imports --- #
# Import your MNISTDataset from dataset.py
# Import your TransformerClassifier from model.py
# ------------------ #
# import itertools # Optional: if cycling through test loader
# import wandb # Optional: for logging

if __name__ == "__main__":
    # --- Configuration --- #
    # 1. Define hyperparameters:
    #    - embedding_dim, num_heads, num_layers (match model definition)
    #    - num_classes = 10
    #    - learning_rate (e.g., 0.001)
    #    - batch_size (e.g., 128)
    #    - num_epochs (e.g., 10)
    # 2. Define device (e.g., "cuda" if torch.cuda.is_available() else "cpu")
    # 3. (Optional) Set random seed: torch.manual_seed(42)
    # --------------------- #
    embedding_dim = 64
    num_heads = 4
    num_layers = 3
    num_classes = 10
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 10
    device = "cpu" # Placeholder

    # --- Data Loading --- #
    # 1. Instantiate training dataset: train_dataset = MNISTDataset(split="train")
    # 2. Instantiate test dataset: test_dataset = MNISTDataset(split="test")
    # 3. Create training DataLoader: train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 4. Create test DataLoader: test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # -------------------- #

    # --- Model, Loss, Optimizer --- #
    # 1. Instantiate the model: model = TransformerClassifier(...hyperparameters...)
    # 2. Move model to the target device: model.to(device)
    # 3. Define the loss function: criterion = nn.CrossEntropyLoss()
    # 4. Define the optimizer: optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # ------------------------------ #

    # --- (Optional) Logging Setup --- #
    # Example: wandb.init(project="mnist-transformer-simple", config={...hyperparameters...})
    # -------------------------------- #

    print(f"Starting training on {device}...")

    # --- Training Loop --- #
    for epoch in range(num_epochs):
        # -- Training Phase -- #
        # 1. Set model to training mode: model.train()
        # 2. Initialize epoch training loss accumulator: epoch_train_loss = []
        # 3. Iterate over batches in train_loader (use tqdm for progress bar if desired):
        #    for i, batch in enumerate(train_loader):
        #        a. Get images and labels: images, labels = batch
        #        b. Move data to device: images, labels = images.to(device), labels.to(device)
        #        c. Zero gradients: optimizer.zero_grad()
        #        d. Forward pass: outputs = model(images)
        #        e. Calculate loss: loss = criterion(outputs, labels)
        #        f. Backward pass: loss.backward()
        #        g. Optimizer step: optimizer.step()
        #        h. Append batch loss: epoch_train_loss.append(loss.item())
        #        i. (Optional) Print batch progress

        # -- Evaluation Phase -- #
        # 1. Set model to evaluation mode: model.eval()
        # 2. Initialize epoch test loss and accuracy accumulators: epoch_test_loss = [], epoch_test_accuracy = []
        # 3. Disable gradient calculation: with torch.no_grad():
        # 4. Iterate over batches in test_loader:
        #    for batch in test_loader:
        #        a. Get images and labels: images, labels = batch
        #        b. Move data to device: images, labels = images.to(device), labels.to(device)
        #        c. Forward pass: outputs = model(images)
        #        d. Calculate loss: loss = criterion(outputs, labels)
        #        e. Calculate accuracy:
        #           - Get predictions: predicted = outputs.argmax(dim=1)
        #           - Compare with labels: correct = (predicted == labels).sum().item()
        #           - Accuracy = correct / images.size(0)
        #        f. Append test loss and accuracy: epoch_test_loss.append(loss.item()), epoch_test_accuracy.append(accuracy)

        # -- Epoch Summary -- #
        # 1. Calculate average train loss, test loss, and test accuracy for the epoch.
        # 2. Print epoch summary: print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: ..., Test Loss: ..., Test Acc: ...")
        # 3. (Optional) Log metrics to wandb: wandb.log({...metrics...})

    print("Training finished.")

    # --- (Optional) Save Model --- #
    # Example: torch.save(model.state_dict(), "mnist_transformer_classifier.pth")
    # ----------------------------- #

"""

# --- Your Actual Python Code Goes Here --- #
# Implement the training script based on the pseudocode above. 