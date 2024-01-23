"""
Welcome to Layer Normalization (LN) practice
By running this code, you will learn how the LN works.
Let's begin!
"""


"""
Step 1
Let's Import the necessary libraries
"""
import torch
import torch.nn as nn


"""
Step 2
Initialize the tokens
"""
input_data = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.5, -0.6], [0.7, 0.8]]])
num_batch, num_tokens, dim = input_data.shape


"""
Step 3
Define the LN layer
"""
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