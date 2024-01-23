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
tokens = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [4.0, 1.0], [5.0, 1.0]]])
num_batch, num_tokens, dim = tokens.shape


"""
Step 3
Define the LN layer
"""
# normalized_shape is the input shape from the last dimension that will be normalized.
layer_norm = nn.LayerNorm(normalized_shape=dim) 


# Manually set gamma (weight) and beta (bias) parameters
gamma = torch.tensor([1.0, 1.5])
beta = torch.tensor([0.0, 0.7])

# Set the weights and biases
layer_norm.weight = nn.Parameter(gamma)
layer_norm.bias = nn.Parameter(beta)


"""
Step 4
Test how things work!
"""
# Apply LayerNorm to the tokens
normalized_tokens = layer_norm(tokens)

# Output the result
print("Original Tokens:\n", tokens, '\n\n')
print("Normalized Tokens:\n", normalized_tokens)