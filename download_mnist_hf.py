import os
from datasets import load_dataset
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_dataset_split(dataset_split, base_dir, split_name):
    """Saves images from a dataset split to a specified directory."""
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    logging.info(f"Saving {split_name} images to {split_dir}...")

    saved_count = 0
    for i, example in enumerate(dataset_split):
        image = example['image']
        label = example['label']
        image_path = os.path.join(split_dir, f"img_{i}_label_{label}.png")
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
    """Main function to download and save the MNIST dataset."""
    output_dir = "assets/mnist"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    try:
        # Load the MNIST dataset from Hugging Face
        logging.info("Loading MNIST dataset from Hugging Face...")
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

        logging.info("Finished processing dataset.")

    except Exception as e:
        logging.error(f"An error occurred during dataset processing: {e}")

if __name__ == "__main__":
    main() 