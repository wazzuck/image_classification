# MNIST Transformer Image Classifier (Simplified)

This guide provides step-by-step instructions to build a simplified Transformer-based image classifier for the MNIST dataset from scratch using PyTorch.

## Project Goal

The goal is to create a model that takes a 28x28 MNIST digit image as input and predicts which digit (0-9) it represents, using a Transformer encoder architecture.

## Steps

1.  **Setup Environment:**
    *   Create a project directory (e.g., `image_classification`).
    *   Set up a Python virtual environment.
    *   Create a `requirements.txt` file listing the necessary libraries (PyTorch, torchvision, datasets).
    *   Install the dependencies: `pip install -r requirements.txt`.

2.  **Create `dataset.py`:**
    *   Implement a PyTorch `Dataset` class to load the MNIST dataset.
    *   Use the `datasets` library to download and access MNIST.
    *   Apply necessary transforms (e.g., `ToTensor`) to convert images to PyTorch tensors.
    *   This script should provide data loaders for both training and testing splits.

3.  **Create `model.py`:**
    *   Define the building blocks of the Transformer encoder:
        *   `ImageEmbedding`: Takes the raw image, divides it into patches (e.g., 4 patches of 14x14), flattens each patch, and projects it to an embedding dimension using a linear layer.
        *   `PositionalEncoding`: Adds positional information to the patch embeddings. This can be learned (using `nn.Embedding`) or fixed (using sine/cosine functions).
        *   `SelfAttentionBlock`: Implements a standard Transformer block containing multi-head self-attention, layer normalization, and a feed-forward network.
        *   `TransformerEncoder`: Stacks multiple `SelfAttentionBlock` layers.
    *   Define the main `TransformerClassifier` model:
        *   It should take the image tensor as input.
        *   Pass the image through `ImageEmbedding`.
        *   Add positional encodings.
        *   Pass the result through the `TransformerEncoder`.
        *   Aggregate the output sequence (e.g., by taking the mean across the patch dimension).
        *   Pass the aggregated representation through a final linear layer (classifier head) to get the logits for the 10 classes.

4.  **Create `train.py`:**
    *   Import the dataset and model classes.
    *   Instantiate the model, loss function (e.g., `nn.CrossEntropyLoss`), and optimizer (e.g., `torch.optim.Adam`).
    *   Create data loaders using the `Dataset` class from `dataset.py`.
    *   Implement the training loop:
        *   Iterate over epochs.
        *   Iterate over batches from the training data loader.
        *   Get images and labels from the batch.
        *   Perform the forward pass: `outputs = model(images)`.
        *   Calculate the loss: `loss = criterion(outputs, labels)`.
        *   Perform the backward pass and optimizer step.
    *   Implement an evaluation loop (within the training loop or separately):
        *   Iterate over batches from the test data loader.
        *   Perform the forward pass in evaluation mode (`model.eval()`, `with torch.no_grad():`).
        *   Calculate test loss and accuracy.
    *   Print or log training/validation metrics (loss, accuracy).
    *   (Optional) Save the trained model's state dictionary.

5.  **Create `evaluate.py` (Optional but Recommended):**
    *   Import the dataset and model classes.
    *   Load a pre-trained model checkpoint.
    *   Instantiate the test data loader.
    *   Run the evaluation loop (similar to the one in `train.py`) on the test set.
    *   Print the final test accuracy and loss.

6.  **Run Training:**
    *   Execute the training script: `python train.py`.
    *   Monitor the output for loss and accuracy improvements.

7.  **Run Evaluation:**
    *   If you implemented `evaluate.py`, run it after training: `python evaluate.py`.

This structured approach will guide you through implementing each component of the MNIST Transformer classifier. 