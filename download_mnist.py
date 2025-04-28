# Import the datasets library
from datasets import load_dataset

# MNIST dataset name on Hugging Face Hub
dataset_name = "mnist"

print(f"Downloading {dataset_name} dataset from Hugging Face...")
try:
    # Load the dataset. This will download it if not already in cache.
    # MNIST is small and doesn't require authentication.
    dataset = load_dataset(dataset_name)
    print("Dataset downloaded successfully!")

    # Optional: Print dataset information
    print("\nDataset information:")
    print(dataset)

    # Optional: See available splits (usually 'train' and 'test' for MNIST)
    print("\nAvailable splits:", list(dataset.keys()))

    # Optional: Access a split (e.g., 'train')
    # if 'train' in dataset:
    #     train_split = dataset['train']
    #     print(f"\nNumber of examples in 'train' split: {len(train_split)}")
    #     # Optional: Access a specific example
    #     if len(train_split) > 0:
    #          print("\nFirst example from the 'train' split:")
    #          print(train_split[0]) # Contains 'image' (PIL Image) and 'label' (int)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have the 'datasets' library installed (`pip install datasets`) and an internet connection.") 