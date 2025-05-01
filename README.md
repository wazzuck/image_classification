# MNIST Tiled Image Sequence Predictor (Encoder-Decoder Transformer)

This guide provides step-by-step instructions to build an Encoder-Decoder Transformer model for predicting the sequence of digits in a 2x2 tiled MNIST image, inspired by the original "Attention Is All You Need" paper and the Vision Transformer concept.

## Project Goal

The goal is to create a sequence-to-sequence model that:
1. Takes a 56x56 image containing four MNIST digits arranged in a 2x2 grid as input.
2. **Encodes** the image information using a Transformer Encoder.
3. **Decodes** the encoded representation autoregressively to generate the sequence of the four digits present in the image (e.g., top-left, top-right, bottom-left, bottom-right).

## Steps

1.  **Setup Environment:**
    *   Ensure you are in the `image_classification` directory.
    *   Have a Python virtual environment activated.
    *   Verify `requirements.txt` contains `torch`, `torchvision`, and `datasets`.
    *   Install dependencies if needed: `pip install -r requirements.txt`.

2.  **Create `dataset.py`:**
    *   Implement a PyTorch `Dataset` class (`TiledMNISTSeqDataset`) to generate the training data.
    *   For each sample, randomly select four MNIST digits.
    *   Create a 56x56 tiled image.
    *   Define a vocabulary including digits 0-9 and special tokens like `<start>`, `<end>`, `<pad>`.
    *   The dataset should return:
        *   The 56x56 tiled image tensor.
        *   The **input sequence** for the decoder (e.g., tensor for `<start> digit_tl digit_tr digit_bl digit_br`).
        *   The **target output sequence** for calculating loss (e.g., tensor for `digit_tl digit_tr digit_bl digit_br <end>`).

3.  **Create `model.py`:**
    *   Define the **Encoder** components (similar to ViT encoder):
        *   `ImageEmbedding`: Processes the 56x56 image (e.g., using patches like 14x14 or 7x7) into patch embeddings.
        *   `PositionalEncoding` for image patches.
        *   `TransformerEncoder` (stack of `SelfAttentionBlock`s) to process patch embeddings.
    *   Define the **Decoder** components (similar to original Transformer decoder):
        *   `nn.Embedding` for the output vocabulary (digits + special tokens).
        *   `PositionalEncoding` for the output sequence.
        *   `TransformerDecoderBlock`: Contains Masked Self-Attention, Cross-Attention (with Encoder output), Feed-Forward, Add & Norm layers.
        *   `TransformerDecoder` (stack of `TransformerDecoderBlock`s).
    *   Define the main `EncoderDecoderTransformer` model:
        *   Takes the image and the decoder input sequence.
        *   Passes the image through the Encoder.
        *   Passes the input sequence and the Encoder output through the Decoder.
        *   Applies a final linear layer to map Decoder output to vocabulary logits.

4.  **Create `train.py`:**
    *   Import the `TiledMNISTSeqDataset` and `EncoderDecoderTransformer`.
    *   Instantiate the model, loss function (`nn.CrossEntropyLoss`, ignoring padding), and optimizer.
    *   Create data loaders.
    *   Implement the training loop:
        *   Get the tiled image, decoder input sequence, and target output sequence.
        *   Perform the forward pass: `output_logits = model(image, decoder_input_sequence)`.
        *   Calculate the loss by comparing `output_logits` with the `target_output_sequence` (reshape logits and targets correctly for CrossEntropyLoss, often `[batch_size * seq_len, vocab_size]` vs `[batch_size * seq_len]`).
        *   Perform backward pass and optimizer step.
    *   Implement an evaluation loop (calculating loss on the test set, potentially sequence accuracy).
    *   Print/log metrics.
    *   (Optional) Save the model.

5.  **Create `evaluate.py`:**
    *   Load a pre-trained `EncoderDecoderTransformer` model.
    *   Instantiate the test data loader (may only need images if generating from scratch).
    *   Implement **autoregressive decoding**:
        *   Encode the input image once.
        *   Start with a `<start>` token as the initial decoder input.
        *   In a loop (up to max sequence length):
            *   Pass the current decoder input sequence and encoded image through the decoder.
            *   Get logits for the *next* token.
            *   Select the token with the highest probability (argmax).
            *   Append the predicted token to the decoder input sequence.
            *   If `<end>` token is predicted, stop.
        *   Compare the generated sequence with the true sequence.
    *   Calculate metrics like full sequence accuracy or per-token accuracy.
    *   Optionally display input tiles and their generated digit sequences.

6.  **Run Training:** `python train.py`
7.  **Run Evaluation:** `python evaluate.py`

This structured approach will guide you through implementing each component of the MNIST Tiled Image Sequence Predictor. 