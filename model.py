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

'''
Positional Encoding is used to add positional information to the input sequence. It allows the model to
understand the order and relative position of tokens in the sequence, which is important for tasks like
language modeling and machine translation.

d_model: Dimension of embedding vector. Every token will be represented as a vector of length d_model.
seq_len: The maximum sequence length. The number of positions for which positional encodings are generated.
dropout: Dropout rate to apply to the positional encodings to avoid overfitting.

EXAMPLE WALKTHROUGH:
seq_len = 10, d_model = 6.
Input tensor x has shape: [batch_size=2, seq_len=10, d_model=6]

Shape of PE is [1, seq_len=10, d_model=6]

Adds PE to the input x element_wise
Ouput has the same shape as x
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) filled with zeros. Stores positional encodings for each position.
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) with values [0, 1, 2, ..., seq_len - 1] (in 1 column). Each row corresponds to a position in the sequence.
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # .unsqueeze(1) essentially makes it a column vector.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model) allows positional encoding matrix to be added to inputs of shape [batch_size, seq_len, d_model]

        # Buffers are saved with the model but are not updated during training.
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Ensures positional encodings match sequence length of input tensor and are not trainable.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

'''
Layer Normalization standardizes inputs along feature dimension. Helps to ensure inputs to a layer have
consistent distribution, which improves convergence during training. Helps gradients flow more smoothly avoiding
vanishing/exploding gradients.

eps: A small value added to the denominator to avoid division by zero.
alpha: Trainable param initialized to 1. Used to scale normalized output.
bias: Trainable param initialized to 0. Used to shift normalized output.

Bias + alpha allow model to learn an affine transformation of the normalized input.

EXAMPLE WALKTHROUGH:
x = [[1, 2, 3], [4, 5, 6]]
For first row:
mean = 2
s.d. = 0.8165

Normalized: (x - mean) / (s.d. + bias) = [-1.2247, 0, 1.2247]
y = alpha * (x - mean) / (s.d. + bias) + bias = [-1.2247, 0, 1.2247]
'''
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = 1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

'''
Feed Fowards Block introduces non-linearity and transformation after attention layers
in transformer. Consists of two fully connected layers (Linear) and an activation function (ReLU)
and dropout for regularization.

Formula looks like:
FFN(x)=Linear(Dropout(ReLU(Linear(x))))

EXAMPLE WALKTHROUGH:
Operation	Input Shape	                Output Shape
Input	    (Batch, Seq_Len, d_model)	(Batch, Seq_Len, d_model)
linear_1	(Batch, Seq_Len, d_model)	(Batch, Seq_Len, d_ff)
ReLU	    (Batch, Seq_Len, d_ff)	    (Batch, Seq_Len, d_ff)
Dropout	    (Batch, Seq_Len, d_ff)	    (Batch, Seq_Len, d_ff)
linear_2	(Batch, Seq_Len, d_ff)	    (Batch, Seq_Len, d_model)

'''
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # output1 = W1 * x + B1
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout) # Randomly sets some elements of tensor to 0 to prevent overfitting.
        # output2 = W2 * ReLU(output1) + B2
        self.linear_2 = nn.Linear(d-ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))