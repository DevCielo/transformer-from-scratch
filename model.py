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


'''
Multihead attention allows models to focus on different parts of sequences (or multiple sequences) 
simultaneously by using multiple attention heads. It processes queries, keys, and values and calculates
attention over the sequence.

Query (Q): What we are searching for
Key (K): What we are comparing against
Value (V): Information extracted

Attention computs a weighted sum of values based on similarity between queries and keys.
Similarity measured using dot product between query and key, scaled by sqrt(dk).

Instead of single attention. Embedding space is split into h heads, 
compute attention sperately for each head then combine results.

EXAMPLE WALKTHROUGH:
Assume:
Batch = 2, Seq_Len = 5, d_model = 512, h = 8

Step	                    Shape
Input (q, k, v)	            (2, 5, 512)
After w_q, w_k, w_v	        (2, 5, 512)
Reshape for multi-head	    (2, 5, 8, 64) → (2, 8, 5, 64)
Attention scores	        (2, 8, 5, 5)
Weighted sum (x)	        (2, 8, 5, 64)
Combine heads (transpose)	(2, 5, 8, 64) → (2, 5, 512)
Final output (w_o(x))	    (2, 5, 512)
'''
class MultiHeadAttentionBlock(nn.Module):

    def __init(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # number of attention heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h # dimension of each attention head
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) # Combines all attention heads back into original dimensions.
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Masking is used to prevent attetion to certain positions (e.g. padding tokens or future tokens)
        if mask is not None:
            # masked_fill_ replaces masked positions with -infinity, making softmax values 0.
            attention_scores.masked_fill_(mask == 0, -1e9)
        # Normalizes attention scores to sum to 1 across sequence length
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, Seq_Len, Seq_Len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    # Mask is a tensor of shape (batch_size, seq_len, seq_len)
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        # Split embedding space into h heads by reshaping.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calls attention method to compute weighted sums and attention scores.
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        # Combine attention heads
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

'''
A residual connection adds the input x of a layer directly to its output sublayer(x). 
Helps address problems like vanishing gradient and allows deeper models to be trained effectively.

Residual connections are combined with layer normalization to stabilize training.

Essentially creates a skip connection by allowing the origina input to a layer to be added 
to its output.
'''
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        #  Normalizes input to sublayer then applies dropout and adds this to the input (giving like a skip connection)
        return x + self.dropout(sublayer(self.norm(x)))

'''
Encoder block represents a single layer in the transformer encoder. It consists of a 
multi-head self-attention clock, a feed-forward block, and two residual connections.

The first residual connection is between the self-attention block and its input.
The second is between the feed-forward block and its input.

In the forward pass. The residual connection normalizes x before passing it to self-attention block.
x is used as Q, K, V in the attention mechanism. src_mask ensures attention is restricted to valid positions.
The original x is added to the output of the self-attention block.

The x is normalized before passing into feed-forward block. 
A non-linear transformation is independently applied to each position in x.
Added original x to the output of feed-forward block.
'''
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Creates a module list of size 2
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # the first position is a residual connection between the multi-head attention block and the input.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) 
        # the second position is a connection between the feed forward block and the input.
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
'''
Consists of multiple encoder blocks stacked together with a final layer normalization at the end.
Essentially represents the entire transformer code.
'''
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) 
        return self.norm(x) 