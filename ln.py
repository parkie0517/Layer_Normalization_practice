"""
Welcome to Layer Normalization (LN) practice
By running this code, you will learn how the LN works.
Let's begin!
"""

# Step 1
"""
Let's Import the necessary libraries
"""
import torch
import torch.nn as nn


# Step 2
"""Define the number of tokens and its dimension"""
num_tokens = 4 # number of tokens
dim = 2 # embedding size of the tokens

# Step 3
"""Initialize the tokens"""
tokens = torch.randn(num_tokens, dim)

# Step 4
"""Define the LN layer"""
# normalized_shape is the input shape from the last dimension onwards that will be normalized.
layer_norm = nn.LayerNorm(normalized_shape=dim)

# Optionally, set custom gamma (scale) and beta (shift) parameters
# Here, both gamma and beta are of shape (2,) as our last dimension is of size 2
gamma = torch.randn(dim)
beta = torch.randn(dim)

# Manually set the weights and biases
layer_norm.weight = nn.Parameter(gamma)
layer_norm.bias = nn.Parameter(beta)

# Apply LayerNorm to the tokens
normalized_tokens = layer_norm(tokens)

# Output the result
print("Original Tokens:\n", tokens)
print("Normalized Tokens:\n", normalized_tokens)