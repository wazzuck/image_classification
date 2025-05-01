# Purpose: Evaluate a trained Encoder-Decoder Transformer using
#          autoregressive decoding to predict digit sequences from tiled MNIST images.

# Pseudocode:
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
# --- Your Imports --- #
# Import TiledMNISTSeqDataset (including vocab info: start/end/pad indices, index_to_label, max_seq_len)
# Import EncoderDecoderTransformer, generate_square_subsequent_mask
# ------------------ #
# import tqdm


def greedy_decode(model, image, max_len, start_symbol_idx, device):
    model.eval()
    # --- Your Code --- #
    # 1. Move image to device.
    # 2. Encode the image once: `memory = model.encode(image)` # memory shape: [1, num_patches, emb_dim]
    # 3. Initialize decoder input sequence: `ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)` # Shape: [1, 1]
    # 4. Loop from 1 to max_len-1:
    #    a. Move memory to device.
    #    b. Generate target mask for current sequence length: `tgt_mask = generate_square_subsequent_mask(ys.size(1), device).type(torch.bool)`
    #    c. Decode: `out = model.decode(ys, memory, tgt_mask)` # out shape: [1, current_seq_len, text_emb_dim]
    #    d. Get logits for the *last* token: `prob = model.classifier(out[:, -1])` # prob shape: [1, vocab_size]
    #    e. Get prediction: `_, next_word_idx = torch.max(prob, dim=1)` # next_word_idx shape: [1]
    #    f. Convert index tensor to integer: `next_word = next_word_idx.item()`
    #    g. Append prediction to decoder input sequence: `ys = torch.cat([ys, torch.ones(1, 1).type_as(image.data).fill_(next_word)], dim=1)`
    #    h. If next_word is the end_symbol_idx, break the loop.
    # 5. Return the generated sequence (excluding the start token): `ys[0, 1:]`
    # ---------------- #
    return None # Placeholder

if __name__ == "__main__":
    # --- Configuration --- #
    # 1. Define model hyperparameters (must match saved model).
    # 2. Define device.
    # 3. Specify checkpoint path.
    # 4. Get vocab info from dataset: start_idx, end_idx, pad_idx, index_to_label, max_seq_len, vocab_size
    # --------------------- #
    # ... (similar to train.py config) ...
    model_checkpoint_path = "path/to/your/encoder_decoder_model.pth" # <<< CHANGE THIS
    start_token_idx = 10 # Placeholder
    end_token_idx = 11   # Placeholder
    max_seq_len = 6      # Placeholder
    device = "cpu"       # Placeholder

    # --- Data Loading --- #
    # 1. Instantiate test dataset: test_dataset = TiledMNISTSeqDataset(split="test")
    # 2. Create DataLoader (batch_size=1 often easiest for sequential generation/evaluation, but >1 is possible).
    #    test_loader = DataLoader(test_dataset, batch_size=1)
    # -------------------- #

    # --- Model Initialization and Loading --- #
    # 1. Instantiate model: model = EncoderDecoderTransformer(...hyperparameters...)
    # 2. Load state dict.
    # 3. Move model to device.
    # 4. Set to evaluation mode: model.eval()
    # --------------------------------------- #

    print(f"Starting evaluation with greedy decoding on {device}...")

    # --- Evaluation Loop --- #
    total_sequences = 0
    correct_sequences = 0
    with torch.no_grad():
        # Iterate through test_loader (use tqdm if desired)
        for batch in test_loader:
            # a. Get image, _, target_seq (input_seq not needed for generation).
            # b. Get image tensor (add batch dim if batch_size=1): image = image.unsqueeze(0) if batch_size == 1 else image
            # c. Generate prediction using greedy_decode helper function:
            #    generated_seq = greedy_decode(model, image, max_seq_len, start_token_idx, device)
            # d. Get the true target sequence (remove start/end/pad tokens for comparison if needed).
            #    true_seq = target_seq[0, :-1] # Example: remove <end> token if target includes it
            # e. Compare generated_seq with true_seq.
            #    - Ensure they have comparable lengths (e.g., truncate generated if too long).
            #    - is_correct = torch.equal(generated_seq, true_seq[:len(generated_seq)])
            # f. Update counts: total_sequences += 1, correct_sequences += 1 if is_correct else 0
            # g. (Optional) Print some examples: 
            #    if total_sequences <= 5:
            #        print(f" Sample {total_sequences}")
            #        print(f"  True: {true_seq.tolist()}")
            #        print(f"  Pred: {generated_seq.tolist()}")
            #        # Convert back to digits using index_to_label for readability

    # --- Calculate Final Metrics --- #
    final_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    # ------------------------------- #

    print(f"\nEvaluation Results:")
    print(f"  Sequence Accuracy: {final_accuracy:.4f}")

"""

# --- Your Actual Python Code Goes Here --- #
# Implement the evaluation script with greedy decoding.
# Remember necessary imports. 