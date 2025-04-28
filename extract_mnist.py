import os
from datasets import load_dataset
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define target directory
TARGET_DIR = "assets/mnist"

def save_dataset_split(dataset_split, base_dir, split_name):
    """Saves images from a dataset split to a specified directory."""
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    logging.info(f"Saving {split_name} images to {split_dir}...")

    saved_count = 0
    for i, example in enumerate(dataset_split):
        image = example['image']
        label = example['label']
        # Use a consistent naming scheme, maybe simpler than the other script
        image_path = os.path.join(split_dir, f"{split_name}_{i:05d}_label_{label}.png")
        try:
            # Ensure image is in RGB or L mode before saving
            if image.mode != 'RGB' and image.mode != 'L':
                image = image.convert('L') # Convert to grayscale if needed
            image.save(image_path)
            saved_count += 1
        except Exception as e:
            logging.error(f"Could not save image {i} for split {split_name}: {e}")

    logging.info(f"Successfully saved {saved_count} images for the {split_name} split.")

def main():
    """Main function to download MNIST using datasets and save as images."""
    output_dir = TARGET_DIR # Use the defined TARGET_DIR
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    try:
        # Load the MNIST dataset from Hugging Face
        logging.info("Loading MNIST dataset from Hugging Face...")
        # Specify cache_dir to potentially control download location if needed
        # mnist_dataset = load_dataset("mnist", cache_dir=os.path.join(output_dir, ".hf_cache"))
        mnist_dataset = load_dataset("mnist")
        logging.info("Dataset loaded successfully.")

        # Save the training split
        if 'train' in mnist_dataset:
            save_dataset_split(mnist_dataset['train'], output_dir, 'train')
        else:
            logging.warning("Training split not found in the dataset.")

        # Save the test split
        if 'test' in mnist_dataset:
            save_dataset_split(mnist_dataset['test'], output_dir, 'test')
        else:
            logging.warning("Test split not found in the dataset.")

        logging.info(f"Finished processing dataset. Images saved in {output_dir}/train and {output_dir}/test")

    except Exception as e:
        # Use logging for errors instead of print to stderr
        logging.error(f"An error occurred during dataset processing: {e}")
        # Exit with error code if needed, uncomment sys import if using sys.exit
        # import sys
        # sys.exit(1)


if __name__ == "__main__":
    main() 