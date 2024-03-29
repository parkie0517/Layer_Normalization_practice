"""
Welcome to Layer Normalization (LN) practice
By following this code, you will learn how the LN works.
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
The token will have a shape like this -> (1, 2, 4)
Where, (1, 2, 4) = (number of samples in a mini-batch, number of tokens, embedding size of the token)
"""
tokens = torch.tensor([[[3.0, -4.0, -1.0, 2.0],[3.0, 0.0, 5.0, 4.0]]])
num_batch, num_tokens, embedding_dim = tokens.shape
print("Token shape\n"+ f"({num_batch}, {num_tokens}, {embedding_dim})")


"""
Step 3
Define the LN layer
"""
layer_norm = nn.LayerNorm(normalized_shape=embedding_dim) # normalized_shape should be set to embedding size

# Manually set gamma (weight) and beta (bias) parameters
gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
beta = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)

# Set the weights and biases
layer_norm.weight = nn.Parameter(gamma)
layer_norm.bias = nn.Parameter(beta)


"""
Step 4
Test how things work!
"""
# Apply LayerNorm to the tokens
normalized_tokens = layer_norm(tokens)
print("Before LN:\n", tokens, '\n')
print("After LN:\n", normalized_tokens, '\n')



"""
Step 5
Let's try implementing our own LN layer!
Follow the following steps.
    1. Normalize the tokens individually
        - Normalizing means to change the distribution of the data to a standard normal distribution
        - Where (mean = 0, variance = 1)
    2. Multiply the result with gamma and add beta
"""
# 1. Normalize the tokens individually
x= tokens
mean = x.mean(-1, keepdim=True) # calculate the mean of the tokens individually
print("mean:\n", mean, '\n')

var = x.var(-1, unbiased=False, keepdim=True) #unbiased=False option is for calculating the sample variance
print("variance\n", var, '\n')

norm_data = (x - mean)/torch.sqrt(var + 1e-12) # 1e-12 is the epsilon value

# 2. Multiply the normalized data with gamma and add beta
out = gamma*norm_data + beta
print("Custom LN:\n", out)