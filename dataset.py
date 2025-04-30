# Purpose: Define a PyTorch Dataset for loading the standard MNIST dataset.
# This involves fetching the dataset (e.g., using the 'datasets' library)
# and applying necessary transformations (like converting images to tensors).

# Pseudocode:
"""
import torch
from torch.utils.data import Dataset
import datasets
import torchvision.transforms as transforms

class MNISTDataset(Dataset):
    def __init__(self, split="train"):
        # --- Your Code --- #
        # 1. Load the MNIST dataset using the 'datasets' library.
        #    Specify the correct split ('train' or 'test').
        #    Example: self.dataset = datasets.load_dataset("mnist")[split]
        # ---------------- #

        # --- Your Code --- #
        # 2. Define the image transformation.
        #    At minimum, convert images to PyTorch tensors.
        #    Example: self.transform = transforms.ToTensor()
        # ---------------- #
        pass

    def __len__(self):
        # --- Your Code --- #
        # 1. Return the total number of samples in the loaded dataset split.
        #    Example: return len(self.dataset)
        # ---------------- #
        return 0 # Placeholder

    def __getitem__(self, idx):
        # --- Your Code --- #
        # 1. Get the image and label at the given index 'idx' from self.dataset.
        #    Example: image = self.dataset[idx]['image']
        #    Example: label = self.dataset[idx]['label']
        # ---------------- #

        # --- Your Code --- #
        # 2. Apply the transformation to the image.
        #    Example: image_tensor = self.transform(image)
        # ---------------- #

        # --- Your Code --- #
        # 3. Ensure the label is a tensor of the correct type (e.g., torch.long).
        #    Example: label_tensor = torch.tensor(label, dtype=torch.long)
        # ---------------- #

        # --- Your Code --- #
        # 4. Return the transformed image tensor and the label tensor.
        #    Example: return image_tensor, label_tensor
        # ---------------- #
        pass # Placeholder

# Optional: Add a main block to test the dataset
# if __name__ == "__main__":
#    train_dataset = MNISTDataset(split="train")
#    test_dataset = MNISTDataset(split="test")
#    print(f"Train dataset size: {len(train_dataset)}")
#    print(f"Test dataset size: {len(test_dataset)}")
#    img, lbl = train_dataset[0]
#    print(f"First image shape: {img.shape}")
#    print(f"First label: {lbl}")
"""

# --- Your Actual Python Code Goes Here --- #
# Implement the MNISTDataset class based on the pseudocode above. 