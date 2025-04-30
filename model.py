# Purpose: Define the Transformer model architecture for MNIST classification.
# This includes the image embedding layer, positional encoding,
# transformer encoder blocks, and the final classification head.

# Pseudocode:
"""
import torch
from torch import nn
import math # Potentially for fixed positional encoding

# --- Image Embedding --- #
class ImageEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=14, embedding_dim=64):
        super().__init__()
        # --- Your Code --- #
        # 1. Calculate the number of patches (e.g., (img_size // patch_size) ** 2).
        # 2. Define a linear layer (`nn.Linear`) to project the flattened patch 
        #    (size: patch_size * patch_size) to the embedding_dim.
        #    Store img_size, patch_size, num_patches, and the projection layer.
        # ---------------- #
        pass

    def forward(self, x):
        # Input x shape: (batch_size, 1, img_size, img_size)
        # --- Your Code --- #
        # 1. Reshape/view the input image `x` into patches.
        #    - Use `x.unfold` or manual slicing/reshaping.
        #    - Target shape might be (batch_size, num_patches, patch_size*patch_size).
        # 2. Apply the linear projection layer to the patches.
        # 3. Return the resulting patch embeddings.
        #    - Output shape: (batch_size, num_patches, embedding_dim)
        # ---------------- #
        pass # Placeholder

# --- Positional Encoding --- #
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=10): # max_len should accommodate num_patches
        super().__init__()
        # --- Your Code (Option 1: Learned Encoding) --- #
        # 1. Define an `nn.Embedding` layer.
        #    - Input dimension: max_len (needs to be >= number of patches).
        #    - Output dimension: embedding_dim.
        # --- Your Code (Option 2: Fixed Sine/Cosine Encoding) --- #
        # 1. Create a fixed positional encoding matrix (PE) of shape (max_len, embedding_dim).
        #    - Use the standard sine/cosine formula based on position and dimension.
        # 2. Register it as a buffer (`self.register_buffer('pe', pe)`).
        # ---------------- #
        pass

    def forward(self, x):
        # Input x shape: (batch_size, num_patches, embedding_dim)
        # --- Your Code (Option 1: Learned Encoding) --- #
        # 1. Create position indices (0 to num_patches-1).
        # 2. Pass indices through the embedding layer.
        # 3. Add the positional embeddings to the input `x`.
        # --- Your Code (Option 2: Fixed Sine/Cosine Encoding) --- #
        # 1. Add the precomputed positional encoding matrix (sliced to num_patches) to `x`.
        # ---------------- #
        # Return shape: (batch_size, num_patches, embedding_dim)
        return x # Placeholder

# --- Self-Attention Block --- #
class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        # --- Your Code --- #
        # 1. Define the Multi-Head Self-Attention layer (`nn.MultiheadAttention`).
        #    - Set `embed_dim=embedding_dim`, `num_heads=num_heads`, `batch_first=True`.
        # 2. Define the first Layer Normalization (`nn.LayerNorm`).
        # 3. Define the Feed-Forward Network (FFN):
        #    - Usually `Linear -> ReLU -> Linear`.
        #    - The intermediate dimension is often 4 * embedding_dim.
        # 4. Define the second Layer Normalization.
        # ---------------- #
        pass

    def forward(self, x):
        # Input x shape: (batch_size, num_patches, embedding_dim)
        # --- Your Code --- #
        # 1. Apply Multi-Head Self-Attention.
        #    - Note: `nn.MultiheadAttention` expects query, key, value. For self-attention, they are all `x`.
        #    - `attn_output, _ = self.multi_head_attention(x, x, x)`
        # 2. Add the input `x` to the attention output (Residual Connection) and apply the first Layer Normalization.
        #    - `x = self.norm1(x + attn_output)`
        # 3. Pass the result through the Feed-Forward Network.
        #    - `ff_output = self.feed_forward(x)`
        # 4. Add the result from step 2 to the FFN output (Residual Connection) and apply the second Layer Normalization.
        #    - `x = self.norm2(x + ff_output)`
        # 5. Return the result.
        #    - Output shape: (batch_size, num_patches, embedding_dim)
        # ---------------- #
        return x # Placeholder

# --- Transformer Encoder --- #
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        # --- Your Code --- #
        # 1. Create a `nn.ModuleList` containing `num_layers` instances of `SelfAttentionBlock`.
        # ---------------- #
        pass

    def forward(self, x):
        # Input x shape: (batch_size, num_patches, embedding_dim)
        # --- Your Code --- #
        # 1. Iterate through the layers in the ModuleList, passing the output of one layer
        #    as the input to the next.
        # 2. Return the output of the final layer.
        #    - Output shape: (batch_size, num_patches, embedding_dim)
        # ---------------- #
        return x # Placeholder

# --- Transformer Classifier --- #
class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_classes, img_size=28, patch_size=14):
        super().__init__()
        # --- Your Code --- #
        # 1. Instantiate `ImageEmbedding`.
        # 2. Instantiate `PositionalEncoding` (ensure max_len >= num_patches).
        # 3. Instantiate `TransformerEncoder`.
        # 4. Define the final classification head (`nn.Linear`).
        #    - Input dimension: embedding_dim.
        #    - Output dimension: num_classes (10 for MNIST).
        # ---------------- #
        pass

    def forward(self, x):
        # Input x shape: (batch_size, 1, img_size, img_size)
        # --- Your Code --- #
        # 1. Pass input `x` through `ImageEmbedding`.
        #    - Result shape: (batch_size, num_patches, embedding_dim)
        # 2. Add positional encodings to the result.
        # 3. Pass the result through the `TransformerEncoder`.
        # 4. Aggregate the output sequence.
        #    - Option A: Take the output corresponding to a special [CLS] token (if added).
        #    - Option B: Calculate the mean of the patch embeddings along the sequence dimension (dim=1).
        #    - Result shape: (batch_size, embedding_dim)
        # 5. Pass the aggregated representation through the final classification head.
        # 6. Return the logits.
        #    - Output shape: (batch_size, num_classes)
        # ---------------- #
        pass # Placeholder

# Optional: Add a main block to test the model dimensions
# if __name__ == "__main__":
#    dummy_img = torch.randn(4, 1, 28, 28) # Batch of 4 images
#    model = TransformerClassifier(embedding_dim=64, num_heads=4, num_layers=3, num_classes=10)
#    output = model(dummy_img)
#    print(f"Input shape: {dummy_img.shape}")
#    print(f"Output shape: {output.shape}") # Should be [4, 10]
"""

# --- Your Actual Python Code Goes Here --- #
# Implement the classes based on the pseudocode above. 