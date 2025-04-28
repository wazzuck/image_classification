# SELF-ATTENTION MECHANISM: A NUMPY IMPLEMENTATION
# ===============================================
# This script demonstrates the core calculation of the self-attention mechanism,
# a key component of Transformer models, using Python and NumPy.
# Self-attention allows a model to weigh the importance of different words in a
# sequence when processing each word, enabling it to capture contextual relationships.
# It dynamically decides which parts of the sequence are relevant for understanding
# a specific word, overcoming limitations of fixed-length context vectors in older models.

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x.

    Softmax converts a vector of raw scores (logits) into a probability distribution,
    where each element is between 0 and 1, and all elements sum to 1.
    This is crucial for attention weights, as they represent how much focus
    to put on each input element.

    Attention weights are the core mechanism that allows a model to dynamically focus
    on different parts of the input sequence. They determine how much each input element
    contributes to the output representation. For example, when processing the word "bank"
    in "I went to the bank to deposit money", the attention weights would be higher for
    words like "deposit" and "money" compared to "went" or "to", helping the model understand
    the financial context rather than a river bank.

    In our case, we're using it to decide how much attention to pay to different parts of a sentence.
    Higher scores get more attention, but everything adds up to 100% attention total.

    Think of it like a voting system where we want to convert raw scores into percentages.
    For example, if three students got test scores of 90, 80, and 70, softmax would convert these
    into percentages like 60%, 30%, and 10% that add up to 100%. This helps us understand the
    relative importance of each score.

    For example, if three students got test scores of 90, 80, and 70, softmax would:
    1. Subtract the maximum score (90) from each score to get: 0, -10, -20
    2. Exponentiate these values: e^0=1, e^-10≈0.000045, e^-20≈0.000000002
    3. Sum these values: 1 + 0.000045 + 0.000000002 ≈ 1.000045
    4. Divide each value by the sum to get percentages: 99.995%, 0.0045%, 0.0002%

    Example scores and their attention weights after softmax:
    | Raw Score | After Max Subtraction | After Exponentiation | Final Attention Weight |
    |-----------|----------------------|----------------------|----------------------|
    | 90        | 0                    | 1.0000               | 99.995%              |
    | 80        | -10                  | 0.000045             | 0.0045%              |
    | 70        | -20                  | 0.000000002          | 0.0002%              |

    This helps us understand the relative importance of each score, with the highest score getting almost all the attention.

    Args:
        x (np.ndarray): Input array of scores. Softmax is applied along the last axis.

    Returns:
        np.ndarray: Array of the same shape as x, containing probabilities.
    """
    # Subtracting the max value before exponentiating enhances numerical stability,
    # preventing potential overflow/underflow issues with very large/small exponents.
    # This is like saying "let's measure everything relative to the highest score"
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    # Divide each exponentiated score by the sum of all exponentiated scores in its row/group.
    # This ensures everything adds up to 100%
    return e_x / e_x.sum(axis=-1, keepdims=True)
# --- Step 1: Input Embeddings ---
# Input data needs to be represented numerically. In NLP, words are typically converted
# into dense vectors called embeddings. These embeddings capture semantic meaning,
# such that similar words have similar vectors.
# Here, we define embeddings for a short sequence: "Self attention example".

# Represent our input sentence: "Self attention example"
# Let's assume each word is represented by a 4-dimensional embedding vector (embedding_dim = 4).
# Sequence length (number of words) is 3.
# Shape: (sequence_length, embedding_dim)
embeddings = np.array([
  [1, 0, 1, 0], # Embedding for "Self"
  [0, 1, 0, 1], # Embedding for "attention"
  [1, 1, 1, 1]  # Embedding for "example"
])

print("Input Embeddings (Shape: {embeddings.shape})")
print(embeddings)
print("-" * 30)

# --- Step 2: Initialize Weight Matrices ---
# Self-attention uses three learned weight matrices (Wq, Wk, Wv) to project the input
# embeddings into three different spaces: Query (Q), Key (K), and Value (V).
# - Query (Q): Represents the current word's perspective, asking "What am I looking for?"
# - Key (K): Represents each word's potential relevance, answering "What information do I hold?"
# - Value (V): Represents the actual content or meaning of each word, answering "What do I actually contribute?"
# These matrices are learned during the model training process.

# Define the dimensions for Q, K, V spaces.
# embedding_dim: Dimension of the input word embeddings.
# d_k: Dimension of the Query and Key vectors. Often set to embedding_dim / num_heads in multi-head attention.
# d_v: Dimension of the Value vectors. Often same as d_k.
embedding_dim = 4
d_k = 3 # Dimension for Keys/Queries
d_v = 2 # Dimension for Values

# Initialize weight matrices randomly. In a real Transformer, these weights are learned.
# Shape Wq: (embedding_dim, d_k)
# Shape Wk: (embedding_dim, d_k)
# Shape Wv: (embedding_dim, d_v)
np.random.seed(42) # for reproducibility
W_q = np.random.rand(embedding_dim, d_k)
W_k = np.random.rand(embedding_dim, d_k)
W_v = np.random.rand(embedding_dim, d_v)

print("Weight Matrices (Randomly Initialized - Learned in Practice):")
print(f"W_q shape: {W_q.shape}") # (4, 3)
print(f"W_k shape: {W_k.shape}") # (4, 3)
print(f"W_v shape: {W_v.shape}") # (4, 2)
print("-" * 30)

# --- Step 3: Project Embeddings into Q, K, V spaces ---
# Multiply the input embeddings matrix by each weight matrix to get Q, K, and V matrices.
# Q = Embeddings @ W_q  -> Shape: (seq_length, d_k)
# K = Embeddings @ W_k  -> Shape: (seq_length, d_k)
# V = Embeddings @ W_v  -> Shape: (seq_length, d_v)
Q = np.matmul(embeddings, W_q)
K = np.matmul(embeddings, W_k)
V = np.matmul(embeddings, W_v)

print("Projected Vectors:")
print(f"Q (Query) shape: {Q.shape}") # (3, 3)
print(f"K (Key) shape: {K.shape}")   # (3, 3)
print(f"V (Value) shape: {V.shape}") # (3, 2)
print("-" * 30)

# --- Step 4: Calculate Attention Scores ---
# The core idea: how relevant is each word (Key) to the current word (Query)?
# This is calculated by taking the dot product between the Query vector of the current word
# and the Key vectors of all words (including itself).
# Scores = Q @ K^T (where K^T is the transpose of K)
# Resulting shape: (seq_length, seq_length)
# scores[i, j] represents the raw relevance score of word j to word i.
scores = np.matmul(Q, K.T)

print(f"Raw Attention Scores (Q @ K.T) shape: {scores.shape}") # (3, 3)
print(scores)
print("Interpretation: scores[i, j] = relevance of input word j to input word i")
print("-" * 30)

# --- Step 5: Scale Scores ---
# Scaling prevents the dot products from becoming too large, especially with high-dimensional
# vectors. Large values could push the softmax function into regions with very small gradients,
# hindering learning. The standard scaling factor is the square root of the dimension of K (d_k).
# Scaled_Scores = Scores / sqrt(d_k)
scaled_scores = scores / np.sqrt(d_k)

print(f"Scaled Attention Scores (Divided by sqrt(d_k={d_k}) = {np.sqrt(d_k):.2f}):")
print(scaled_scores)
print("-" * 30)

# --- Step 6: Apply Softmax ---
# Convert the scaled scores into probabilities (attention weights). The softmax function
# is applied row-wise, so the weights for each Query word (each row) sum to 1.
# Attention_Weights = softmax(Scaled_Scores)
# Shape: (seq_length, seq_length)
# attention_weights[i, j] is the proportion of attention word i pays to word j.
attention_weights = softmax(scaled_scores)

print(f"Attention Weights (Softmax applied row-wise) shape: {attention_weights.shape}") # (3, 3)
print(attention_weights)
print("Note: Each row sums to 1. Represents attention distribution for each query word.")
print("-" * 30)

# --- Step 7: Calculate Final Output ---
# The final output for each word is a weighted sum of all the Value (V) vectors in the sequence.
# The weights used are the attention weights calculated in the previous step.
# Output = Attention_Weights @ V
# Shape: (seq_length, d_v)
# output[i] is the new representation for word i, incorporating context from the
# entire sequence based on the attention weights.
output = np.matmul(attention_weights, V)

print(f"Final Self-Attention Output (Weighted Sum of Values) shape: {output.shape}") # (3, 2)
print(output)
print("-" * 30)

# --- Summary and Interpretation ---
print("Summary:")
print(f"Original Input Embeddings shape: {embeddings.shape}") # (3, 4)
print(f"Final Output Representation shape: {output.shape}")  # (3, 2)
# Note: The output dimension is d_v. In a full Transformer, often d_v = embedding_dim,
# or the output is projected back to embedding_dim after multi-head attention.

# Let's examine the output for the first word ("Self")
print("\nDetailed Example: Output vector for 'Self' (word 0):", output[0])
print("This vector was derived by weighting the Value vectors (V) based on the attention weights for 'Self' (row 0 of attention_weights):")
print(f"Output[0] = (AttentionWeights[0,0] * V[0]) + (AttentionWeights[0,1] * V[1]) + (AttentionWeights[0,2] * V[2])")
print(f"         = ({attention_weights[0,0]:.2f} * {V[0]}) + ({attention_weights[0,1]:.2f} * {V[1]}) + ({attention_weights[0,2]:.2f} * {V[2]}) ")
print("The resulting vector for 'Self' now incorporates information from 'attention' and 'example', weighted by how relevant they were deemed by the attention mechanism.")
