# Purpose: Define a PyTorch Dataset for generating 2x2 tiled MNIST images
#          and corresponding input/output sequences for an Encoder-Decoder model.

import torch
from torch.utils.data import Dataset
import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F # For padding

# --- Vocabulary Definition --- #
label_to_index = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "<start>": 10, "<end>": 11, "<pad>": 12
}
index_to_label = {v: k for k, v in label_to_index.items()}
vocab_size = len(label_to_index)
start_token_idx = label_to_index["<start>"]
end_token_idx = label_to_index["<end>"]
pad_token_idx = label_to_index["<pad>"]
# Sequence: <start> d1 d2 d3 d4 <end>
max_seq_len = 6

class TiledMNISTSeqDataset(Dataset):
    def __init__(self, split="train"):
        self.base_dataset = datasets.load_dataset("mnist", split=split)
        self.split = split
        self.transform = transforms.ToTensor()
        # Store vocab info
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.pad_token_idx = pad_token_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Select 4 random indices and get their images/labels
        indices = torch.randint(0, len(self.base_dataset), (4,)).tolist()
        images = []
        labels_list = []
        for index in indices:
            item = self.base_dataset[index]
            # 2. Apply transform to images
            images.append(self.transform(item['image']))
            # 4. Get integer labels
            labels_list.append(item['label']) # These are already 0-9

        # 3. Create the 56x56 tiled image tensor
        tile = torch.cat([torch.cat(images[0:2], dim=2), torch.cat(images[2:4], dim=2)], dim=1)

        # 5. Create the decoder input sequence list
        input_seq_list = [self.start_token_idx] + labels_list

        # 6. Create the target output sequence list
        target_seq_list = labels_list + [self.end_token_idx]

        # 7. Convert lists to tensors
        input_seq_tensor = torch.tensor(input_seq_list, dtype=torch.long)
        target_seq_tensor = torch.tensor(target_seq_list, dtype=torch.long)

        # 8. Pad sequences to max_seq_len
        # Input sequence needs padding up to max_len (it's length 5 currently)
        pad_len_input = self.max_seq_len - len(input_seq_tensor)
        if pad_len_input > 0:
             # Pad input sequence (usually only needs 1 pad for <end> if target is used directly)
            input_seq_tensor = F.pad(input_seq_tensor, (0, pad_len_input), value=self.pad_token_idx)

        # Target sequence needs padding up to max_len (it's length 5 currently)
        pad_len_target = self.max_seq_len - len(target_seq_tensor)
        if pad_len_target > 0:
            target_seq_tensor = F.pad(target_seq_tensor, (0, pad_len_target), value=self.pad_token_idx)

        # Ensure sequences do not exceed max_seq_len (safety check)
        input_seq_tensor = input_seq_tensor[:self.max_seq_len]
        target_seq_tensor = target_seq_tensor[:self.max_seq_len]

        # 9. Return the tiled image, input sequence tensor, and target sequence tensor.
        return tile, input_seq_tensor, target_seq_tensor

# --- Main block for testing --- #
if __name__ == "__main__":
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Pad token index: {pad_token_idx}")
    dataset = TiledMNISTSeqDataset(split="train")
    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    img, input_seq, target_seq = dataset[10] # Get 11th item for variety

    print(f"\nSample shapes:")
    print(f"  Image shape: {img.shape}")       # Expected: [1, 56, 56]
    print(f"  Input sequence shape: {input_seq.shape}") # Expected: [max_seq_len] = [6]
    print(f"  Target sequence shape: {target_seq.shape}")# Expected: [max_seq_len] = [6]

    print(f"\nSample sequences (indices):")
    print(f"  Input sequence: {input_seq.tolist()}")
    print(f"  Target sequence: {target_seq.tolist()}")

    # Decode sequences back to labels/tokens using index_to_label
    print(f"\nSample sequences (decoded):")
    print(f"  Input labels: {[index_to_label.get(i.item(), '?') for i in input_seq]}")
    print(f"  Target labels: {[index_to_label.get(i.item(), '?') for i in target_seq]}")

    # Optional: Save the sample tiled image
    from torchvision.transforms.functional import to_pil_image
    import os
    os.makedirs('../assets/image_classification', exist_ok=True)
    try:
        img_pil = to_pil_image(img)
        save_path = '../assets/image_classification/sample_tiled_seq_image.png'
        img_pil.save(save_path)
        print(f"\nSaved sample tiled image to {save_path}")
    except Exception as e:
        print(f"\nCould not save sample image: {e}")