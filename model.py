# Purpose: Define the Encoder-Decoder Transformer architecture for
#          predicting digit sequences from tiled MNIST images.

# Pseudocode:
"""
This file defines the components and the main EncoderDecoderTransformer model
for the image-to-sequence task on tiled MNIST digits.
"""

import torch
from torch import nn
import math

# --- Shared Components (Embeddings, Encodings, Attention Blocks) --
class ImageEmbedding(nn.Module):
    # Processes the 56x56 image into patch embeddings
    def __init__(self, img_size=56, patch_size=14, embedding_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size, embedding_dim)
        # Note: The 'ncan you n.' was likely a typo, removing cls_token initialization for now
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) # Optional CLS token if needed later

    def forward(self, x):
        # Input: (batch_size, 1, 56, 56)
        B, C, H, W = x.shape
        # Ensure input dimensions match configuration
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."

        # Reshape into patches: (B, C, H, W) -> (B, num_patches, patch_size*patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, self.num_patches, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, self.num_patches, -1) # (B, num_patches, patch_size*patch_size)

        # Project patches
        x = self.projection(x)

        # Prepend CLS token (optional, not strictly needed for encoder-decoder seq-to-seq)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # Output: (batch_size, num_patches, embedding_dim)
        return x

class PositionalEncoding(nn.Module):
    # Using learned positional embeddings
    def __init__(self, embedding_dim, max_len=20): # max_len >= max(num_patches, max_seq_len)
        super().__init__()
        # Ensure max_len is large enough for both image patches and text sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embedding_dim))

    def forward(self, x):
        # Input x shape: (batch_size, seq_length, embedding_dim)
        # Add positional embedding (broadcasts along batch dim)
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.size(1):
           raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pos_embedding.size(1)} for PositionalEncoding. Adjust max_len in PositionalEncoding init.")
        return x + self.pos_embedding[:, :seq_len]

# --- Encoder Components ---

class EncoderSelfAttentionBlock(nn.Module):
    # Standard Transformer Encoder block
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Input: (batch_size, num_patches, embedding_dim)
        # Self-Attention
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=mask) # Q, K, V
        x = residual + attn_output

        # FeedForward
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        # Output: (batch_size, num_patches, embedding_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderSelfAttentionBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim) # Final norm after layers

    def forward(self, x, mask=None):
        # Input: (batch_size, num_patches, embedding_dim)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        # Output: (batch_size, num_patches, embedding_dim)
        return x

# --- Decoder Components ---

class DecoderAttentionBlock(nn.Module):
    # Standard Transformer Decoder block
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        # Masked Self-Attention
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        # Cross-Attention (query from decoder, key/value from encoder memory)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embedding_dim)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        # Input x (decoder sequence): (batch_size, seq_len, embedding_dim)
        # Input memory (encoder output): (batch_size, num_patches, embedding_dim)

        # 1. Masked Self-Attention
        residual = x
        x_norm = self.norm1(x)
        self_attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = residual + self_attn_output # Add residual

        # 2. Cross-Attention
        residual = x
        x_norm = self.norm2(x)
        # Query: decoder state (x_norm), Key/Value: encoder memory
        cross_attn_output, _ = self.cross_attention(query=x_norm, key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        x = residual + cross_attn_output # Add residual

        # 3. FeedForward
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x) # Add residual

        # Output: (batch_size, seq_len, embedding_dim)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderAttentionBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim) # Final norm after layers

    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        # Input x: (batch_size, seq_len, embedding_dim)
        # Input memory: (batch_size, num_patches, embedding_dim)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        x = self.norm(x)
        # Output: (batch_size, seq_len, embedding_dim)
        return x

# --- Main Encoder-Decoder Model --- #
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, num_heads,
                 vocab_size, img_embedding_dim, text_embedding_dim,
                 img_size=56, patch_size=14, max_seq_len=6, dropout=0.1):
        super().__init__()

        # Ensure embedding dims are compatible
        assert img_embedding_dim == text_embedding_dim, \
            "Image and text embedding dimensions must be equal for this implementation."
        embedding_dim = img_embedding_dim # Use a single dim name internally

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Determine max length needed for positional encoding
        # It should accommodate both the image patches and the text sequence
        # Add +1 if using CLS token in image embedding, currently commented out
        max_pos_encoding_len = max(self.num_patches, max_seq_len)

        # 1. Image Embedding
        self.image_embedding = ImageEmbedding(img_size, patch_size, embedding_dim)
        # 2. Text Embedding
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        # 3. Positional Encodings (Shared or separate, using separate here)
        self.image_pos_encoding = PositionalEncoding(embedding_dim, max_len=max_pos_encoding_len)
        self.text_pos_encoding = PositionalEncoding(embedding_dim, max_len=max_pos_encoding_len)

        # 4. Transformer Encoder
        self.encoder = TransformerEncoder(embedding_dim, num_heads, num_encoder_layers, dropout)
        # 5. Transformer Decoder
        self.decoder = TransformerDecoder(embedding_dim, num_heads, num_decoder_layers, dropout)

        # 6. Final Linear layer (classifier head)
        self.classifier = nn.Linear(embedding_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self, src_img):
        # Input src_img: (batch_size, 1, 56, 56)
        # 1. Apply Image Embedding
        x = self.image_embedding(src_img) # (batch, num_patches, embed_dim)
        # 2. Add Positional Encoding
        x = self.image_pos_encoding(x)
        x = self.dropout(x)
        # 3. Pass through TransformerEncoder
        memory = self.encoder(x)
        # Output: (batch_size, num_patches, embedding_dim)
        return memory

    def decode(self, tgt_seq, memory, tgt_mask=None):
        # Input tgt_seq (decoder input tokens): (batch_size, seq_len)
        # Input memory (encoder output): (batch_size, num_patches, embedding_dim)
        # 1. Apply Text Embedding to tgt_seq
        tgt_emb = self.text_embedding(tgt_seq) # (batch, seq_len, embed_dim)
        # 2. Add Positional Encoding
        tgt_emb = self.text_pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        # 3. Pass through TransformerDecoder (providing memory and tgt_mask)
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        # Output: (batch_size, seq_len, text_embedding_dim)
        return decoder_output

    def forward(self, src_img, tgt_seq):
        # Input src_img: (batch_size, 1, 56, 56)
        # Input tgt_seq (decoder input tokens): (batch_size, seq_len)

        # 1. Generate target mask for self-attention in decoder
        # Ensures decoder doesn't cheat by looking at future tokens
        tgt_mask = generate_square_subsequent_mask(tgt_seq.size(1), device=tgt_seq.device)

        # 2. Encode the source image
        memory = self.encode(src_img)

        # 3. Decode using target sequence and memory
        decoder_output = self.decode(tgt_seq, memory, tgt_mask=tgt_mask)

        # 4. Apply final linear layer to decoder output to get logits
        logits = self.classifier(decoder_output)

        # Output: (batch_size, seq_len, vocab_size)
        return logits

# Helper to create the mask for the decoder's self-attention
def generate_square_subsequent_mask(sz, device="cpu"):
   # Create a mask with True values on the upper triangle
   mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
   # Fill with -inf where True (don't attend), 0.0 where False (attend)
   # Mask shape will be (sz, sz)
   mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
   return mask

# --- Main block for testing dimensions --- #
if __name__ == "__main__":
    # Define params
    img_emb_dim = 64
    text_emb_dim = 64
    vocab_size = 13 # 0-9 + start/end/pad
    n_layers = 3
    n_heads = 4 # embedding_dim (64) must be divisible by num_heads (4)
    max_len_seq = 6 # Max sequence length for text
    img_s = 56
    patch_s = 14
    batch_size = 4
    seq_len_input = max_len_seq - 1 # Decoder input excludes <end>

    print("Testing EncoderDecoderTransformer dimensions...")
    model = EncoderDecoderTransformer(num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                     num_heads=n_heads, vocab_size=vocab_size,
                                     img_embedding_dim=img_emb_dim, text_embedding_dim=text_emb_dim,
                                     img_size=img_s, patch_size=patch_s, max_seq_len=max_len_seq)

    # Create dummy inputs
    dummy_img = torch.randn(batch_size, 1, img_s, img_s)
    dummy_seq = torch.randint(0, vocab_size, (batch_size, seq_len_input))

    # Perform forward pass
    output_logits = model(dummy_img, dummy_seq)

    print(f"  Input Image Shape: {dummy_img.shape}")
    print(f"  Input Sequence Shape: {dummy_seq.shape}")
    print(f"  Output Logits Shape: {output_logits.shape}")

    # Check output shape
    expected_shape = (batch_size, seq_len_input, vocab_size)
    assert output_logits.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, Got {output_logits.shape}"
    print("\nDimension test passed!")