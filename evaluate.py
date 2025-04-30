# Purpose: Evaluate a trained Transformer Classifier on the MNIST test set.
# This involves loading a saved model checkpoint, loading the test data,
# running the model in evaluation mode, and calculating final metrics.

# Pseudocode:
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
# --- Your Imports --- #
# Import your MNISTDataset from dataset.py
# Import your TransformerClassifier from model.py
# ------------------ #

if __name__ == "__main__":
    # --- Configuration --- #
    # 1. Define hyperparameters (must match the *saved* model):
    #    - embedding_dim, num_heads, num_layers, num_classes
    # 2. Define batch_size.
    # 3. Define device.
    # 4. Specify the path to the saved model checkpoint (.pth or .safetensors).
    #    - model_checkpoint_path = "mnist_transformer_classifier.pth" # Example
    # --------------------- #
    embedding_dim = 64
    num_heads = 4
    num_layers = 3
    num_classes = 10
    batch_size = 128
    device = "cpu" # Placeholder
    model_checkpoint_path = "path/to/your/model.pth" # <<< CHANGE THIS

    # --- Data Loading --- #
    # 1. Instantiate test dataset: test_dataset = MNISTDataset(split="test")
    # 2. Create test DataLoader: test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # -------------------- #

    # --- Model Initialization and Loading --- #
    # 1. Instantiate the model architecture: model = TransformerClassifier(...hyperparameters...)
    # 2. Load the saved state dictionary:
    #    - state_dict = torch.load(model_checkpoint_path, map_location=device)
    #    - model.load_state_dict(state_dict)
    # 3. Move model to device: model.to(device)
    # 4. Set model to evaluation mode: model.eval()
    # --------------------------------------- #

    # --- Evaluation Loop --- #
    # 1. Initialize accumulators: total_loss = 0.0, total_correct = 0, total_samples = 0
    # 2. Define loss function (optional, if you want to report loss): criterion = nn.CrossEntropyLoss()
    # 3. Disable gradient calculation: with torch.no_grad():
    # 4. Iterate over batches in test_loader:
    #    for batch in test_loader:
    #        a. Get images and labels: images, labels = batch
    #        b. Move data to device: images, labels = images.to(device), labels.to(device)
    #        c. Forward pass: outputs = model(images)
    #        d. (Optional) Calculate loss: loss = criterion(outputs, labels)
    #        e. Calculate accuracy:
    #           - Get predictions: predicted = outputs.argmax(dim=1)
    #           - Update totals: total_correct += (predicted == labels).sum().item()
    #                          total_samples += images.size(0)
    #                          if criterion: total_loss += loss.item() * images.size(0)

    # --- Calculate Final Metrics --- #
    # 1. Final accuracy = total_correct / total_samples
    # 2. (Optional) Final average loss = total_loss / total_samples
    # ------------------------------- #

    print(f"Evaluation Results on Test Set:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    # if criterion: print(f"  Average Loss: {final_avg_loss:.4f}")

"""

# --- Your Actual Python Code Goes Here --- #
# Implement the evaluation script based on the pseudocode above. 