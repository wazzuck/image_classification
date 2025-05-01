# MULTI-HEAD ATTENTION MECHANISM: A NUMPY IMPLEMENTATION
# ======================================================
# This script demonstrates the Multi-Head Attention mechanism, an extension of
# self-attention introduced in the "Attention Is All You Need" paper (Transformers).
# Multi-Head Attention allows the model to jointly attend to information from
# different representation subspaces at different positions. Instead of performing
# a single attention function, it projects the queries, keys, and values h times
# with different, learned linear projections. Attention is performed in parallel for
# each projection, yielding h output values. These are concatenated and once again
# projected, resulting in the final values.
# This approach helps the model capture various aspects and relationships in the data.

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x along the last axis."""
    # Subtract max for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Normalize to get probabilities
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --- Input Parameters & Setup ---
# Define the key parameters for the multi-head attention mechanism.

seq_length = 3       # Length of the input sequence (e.g., number of words)
embedding_dim = 6    # Dimension of the input embedding for each element in the sequence
num_heads = 2        # Number of parallel attention heads (h)

# Constraint: The embedding dimension must be divisible by the number of heads.
# This allows the dimension to be split evenly across the heads.
if embedding_dim % num_heads != 0:
    raise ValueError(f"Embedding dimension ({embedding_dim}) must be divisible by the number of heads ({num_heads}).")

# Calculate the dimension of vectors within each attention head.
# d_k_head: Dimension of Query and Key vectors per head.
# d_v_head: Dimension of Value vectors per head.
# Often, d_k_head and d_v_head are set equal (embedding_dim // num_heads).
d_k_head = embedding_dim // num_heads
d_v_head = embedding_dim // num_heads

print(f"Sequence Length: {seq_length}")
print(f"Embedding Dimension (model dim): {embedding_dim}")
print(f"Number of Attention Heads: {num_heads}")
print(f"Dimension per Head (d_k_head / d_v_head): {d_k_head}")
print("-" * 40)

# --- Step 1: Input Embeddings ---
# Start with input embeddings for the sequence.
# Example: "Multi head attention"
# Shape: (seq_length, embedding_dim)
np.random.seed(42) # for reproducibility
embeddings = np.random.rand(seq_length, embedding_dim)

print(f"Input Embeddings (Shape: {embeddings.shape}):")
print(embeddings)
print("-" * 40)

# --- Step 2: Initialize Weight Matrices (Per Head + Output) ---
# Multi-Head Attention requires separate weight matrices for each head
# to project the input embeddings into Q, K, V subspaces for that specific head.
# Additionally, a final weight matrix Wo is needed to project the concatenated
# head outputs back to the desired model dimension.

# List to hold weight matrices for each head
W_q_heads = [] # Query weights per head
W_k_heads = [] # Key weights per head
W_v_heads = [] # Value weights per head

# Initialize weights for each head (learned in practice)
for i in range(num_heads):
    # Shape Wq_i, Wk_i: (embedding_dim, d_k_head)
    W_q_heads.append(np.random.rand(embedding_dim, d_k_head))
    W_k_heads.append(np.random.rand(embedding_dim, d_k_head))
    # Shape Wv_i: (embedding_dim, d_v_head)
    W_v_heads.append(np.random.rand(embedding_dim, d_v_head))

# Final output projection matrix Wo
# Takes the concatenated outputs (size: num_heads * d_v_head) and projects back to embedding_dim.
# Shape Wo: (num_heads * d_v_head, embedding_dim)
W_o = np.random.rand(num_heads * d_v_head, embedding_dim)

print("Weight Matrices (Randomly Initialized - Learned in Practice):")
print(f"Number of Heads: {num_heads}")
print(f"  W_q/W_k shape per head: ({embedding_dim}, {d_k_head})")
print(f"  W_v shape per head: ({embedding_dim}, {d_v_head})")
print(f"W_o shape (output projection): ({num_heads * d_v_head}, {embedding_dim})")
print("-" * 40)

# --- Steps 3 & 4: Calculate Attention For Each Head Independently ---
# The core self-attention calculation (Scaled Dot-Product Attention) is performed
# independently and in parallel for each head.

head_outputs = [] # Store the output of each attention head

print("Calculating Attention for Each Head:")
for i in range(num_heads):
    print(f"\n--- Head {i} ---Processing ---")
    # Get the weight matrices for the current head
    Wq_i = W_q_heads[i]
    Wk_i = W_k_heads[i]
    Wv_i = W_v_heads[i]

    # 3a. Project embeddings into Q, K, V for this specific head
    # Q_i = Embeddings @ Wq_i  -> Shape: (seq_length, d_k_head)
    # K_i = Embeddings @ Wk_i  -> Shape: (seq_length, d_k_head)
    # V_i = Embeddings @ Wv_i  -> Shape: (seq_length, d_v_head)
    Q_i = np.matmul(embeddings, Wq_i)
    K_i = np.matmul(embeddings, Wk_i)
    V_i = np.matmul(embeddings, Wv_i)
    # print(f"Q_{i} shape: {Q_i.shape}")
    # print(f"K_{i} shape: {K_i.shape}")
    # print(f"V_{i} shape: {V_i.shape}")

    # 4a. Calculate scaled dot-product attention scores for this head
    # Scores = (Q_i @ K_i^T) / sqrt(d_k_head)
    scores_i = np.matmul(Q_i, K_i.T) / np.sqrt(d_k_head)
    # print(f"\nScaled Scores_{i} shape: {scores_i.shape}")

    # 4b. Apply softmax to get attention weights for this head
    # attention_weights_i shape: (seq_length, seq_length)
    attention_weights_i = softmax(scores_i)
    # print(f"\nAttention Weights_{i} shape: {attention_weights_i.shape}")

    # 4c. Calculate the weighted sum of Value vectors for this head
    # head_output_i = attention_weights_i @ V_i
    # head_output_i shape: (seq_length, d_v_head)
    head_output_i = np.matmul(attention_weights_i, V_i)
    print(f"Head Output_{i} shape: {head_output_i.shape}")

    # Store the result for this head
    head_outputs.append(head_output_i)

print("-" * 40)

# --- Step 5: Concatenate Head Outputs ---
# Concatenate the outputs from all heads along the feature dimension (last axis).
# This combines the information learned by each independent head.
# Shape: (seq_length, num_heads * d_v_head)
concatenated_heads = np.concatenate(head_outputs, axis=-1)

print(f"Concatenated Head Outputs (Shape: {concatenated_heads.shape}):")
# print(concatenated_heads)
print("-" * 40)

# --- Step 6: Final Linear Transformation (Output Projection) ---
# Apply the final output weight matrix W_o to the concatenated head outputs.
# This projects the combined multi-head information back into the model's
# standard embedding dimension.
# Final Output = Concatenated_Heads @ W_o
# Shape: (seq_length, embedding_dim)
final_output = np.matmul(concatenated_heads, W_o)

print(f"Final Multi-Head Attention Output (Shape: {final_output.shape}):")
print(final_output)
print("-" * 40)

# --- Summary and Interpretation ---
print("Summary:")
print(f"Input Embeddings shape: {embeddings.shape}")
print(f"Number of Heads: {num_heads}")
print(f"Output shape per Head: {(seq_length, d_v_head)}")
print(f"Concatenated Heads shape: {concatenated_heads.shape}")
print(f"Final Output shape (after Wo projection): {final_output.shape}")

print("\nInterpretation:")
print("The multi-head attention mechanism processed the input sequence in parallel across multiple heads.")
print("Each head used its own learned projections (Wq, Wk, Wv) to focus on different aspects or subspaces of the input relationships.")
print("The outputs from these independent heads were combined (concatenated) and then projected back to the original embedding dimension.")
print("This allows the model to capture a richer set of contextual information compared to single-head attention.") 