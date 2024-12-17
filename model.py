import torch
import torch.nn
import math


'''
Input Embeddings are used to map token indices (integers representing words or subwords) to
dense vectors of a specified dimension (d_model), which are used as input to the model. They allow transformers
to process tokens as vectors rather than raw integers.

d_model: Dimension of embedding vector. Every token will be represented as a vector of length d_model.
vocab_size: Total number of unique tokens (words/subwords) in the vocabulary

nn.Embedding: pytorch layer that converts integer indices (token IDs) into dense vectors.
e.g. (vocab_size = 3, d_model = 4)
Token 0 → [0.1, 0.2, -0.1, 0.3]
Token 1 → [0.5, -0.4, 0.3, 0.9]
Token 2 → [-0.2, 0.1, 0.7, -0.3]
Embedding layer learns these vectors during training.

Embedding vectors are scaled by sqrt(d_model). This helps stabilize the input to the transformer by
keeping values at a consistent scale.

EXAMPLE WALKTHROUGH:
x = [[5, 123, 10]] (sequence of 3 words)

output is a tensor of shape [1, 3, 512] (Each word in sequence gets respresented as vector of size 512).
Embedding values are scaled by sqrt(512)
'''

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # size of embedding vector (512 in this case)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Input size [batch_size, sequence_length], Output size [batch_size, sequence_length, d_model]

    def forward(self, x):
        # self.embedding replaces each token ID with its corresponding vector
        return self.embedding(x) * math.srqt(self.d_model) # multiple the weights by sqrt(d_model)