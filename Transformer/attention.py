import numpy as np
from layers.linear import Linear
from layers.activations import softmax

# Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V


# SELF ATTENTION
class Self_Attention:
    
    def __init__(self, embedding_dim):
        self.W_q = Linear(embedding_dim, embedding_dim)
        self.W_k = Linear(embedding_dim, embedding_dim)
        self.W_v = Linear(embedding_dim, embedding_dim)
        self.W_o = Linear(embedding_dim, embedding_dim)

        # store dim of key vector
        self.d_k = embedding_dim

    def forward(self, X):
        Q = self.W_q.forward(X)
        K = self.W_k.forward(X)
        V = self.W_v.forward(X)

        # calculate score
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        scores = softmax(scores) @ V
        return self.W_o.forward(scores)


if __name__ == "__main__":
    np.random.seed(67)
    x = np.random.randn(2, 5, 8)
    attn = Self_Attention(embedding_dim = 8)
    out = attn.forward(x)
    assert out.shape == x.shape
    print(f"attention : {x.shape} -> {out.shape}")


# MULTI-HEAD ATTENTION
class MultiHead_Attention:
    def __init__(self, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads

        # Initialize heads (instances of self attention class)
        self.attention_heads = [Self_Attention(self.head_dim) for _ in range(num_heads)]
        # Initialize W_o
        self.W_o = Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings):
        batch, seq, embedding_dim = embeddings.shape
        num_heads = len(self.attention_heads)
        # Split embeddings into multiple heads
        split = embeddings.reshape(batch, seq, num_heads, self.head_dim)
        split = split.transpose(0, 2, 1, 3)

