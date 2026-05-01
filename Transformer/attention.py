import numpy as np
 
try:
    from .layers.linear import Linear
    from .layers.activations import softmax
except ImportError:
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

    def forward(self, X, mask=None):
        Q = self.W_q.forward(X)
        K = self.W_k.forward(X)
        V = self.W_v.forward(X)

        # calculate score
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        scores = softmax(scores) @ V
        return self.W_o.forward(scores)

# CROSS ATTENTION
class Cross_Attention:

    def __init__(self, embedding_dim):
        self.W_q = Linear(embedding_dim, embedding_dim)
        self.W_k = Linear(embedding_dim, embedding_dim)
        self.W_v = Linear(embedding_dim, embedding_dim)
        self.W_o = Linear(embedding_dim, embedding_dim)

        # store dim of key vector
        self.d_k = embedding_dim

    def forward(self, query_input, context_input, mask=None):
        Q = self.W_q.forward(query_input)
        K = self.W_k.forward(context_input)
        V = self.W_v.forward(context_input)

        # calculate score
        scores = Q @ K.transpose(0,2,1) / np.sqrt(self.d_k)
        scores = softmax(scores) @ V
        return self.W_o.forward(scores)
        

# MULTI-HEAD ATTENTION
class MultiHead_Attention:

    def __init__(self, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Initialize heads (instances of self attention class)
        self.attention_heads = [Self_Attention(self.head_dim) for _ in range(num_heads)]
        # Initialize W_o
        self.W_o = Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings, mask=None):
        batch, seq, embedding_dim = embeddings.shape
        num_heads = len(self.attention_heads)
        # Split embeddings into multiple heads
        split = embeddings.reshape(batch, seq, num_heads, self.head_dim)
        split = split.transpose(0, 2, 1, 3)

        # Apply self attention on each head
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_out = head.forward(split[:, i, :, :])  # (B, S, head_dim)
            head_outputs.append(head_out)

        out = np.stack(head_outputs, axis = 1)          # (B, H, S, head_dim)
        out = out.transpose(0,2,1,3)                    # (B, S, H, head_dim)
        out = out.reshape(batch, seq, self.num_heads * self.head_dim)  # (B, S, D)

        proj = self.W_o.forward(out) # final projection
        return proj

if __name__ == "__main__":
    np.random.seed(67)
    x = np.random.randn(2, 5, 8)
    attn = Self_Attention(embedding_dim = 8)
    out = attn.forward(x)
    assert out.shape == x.shape
    print(f"attention : {x.shape} -> {out.shape}")


    # TEST : FOR Cross Att
    decoder_x = np.random.randn(2, 4, 8)
    encoder_x = np.random.randn(2, 6, 8)
    cross = Cross_Attention(embedding_dim=8)
    out = cross.forward(decoder_x, encoder_x)
    assert out.shape == decoder_x.shape
    print(f"cross attention check passed!")


    # TEST : FOR MLH Att
    x = np.random.randn(2, 5, 8)
    mlh = MultiHead_Attention(embedding_dim=8, num_heads=2)
    out = mlh.forward(x)
    assert out.shape == x.shape
    print(f"multi-head attention check passed!")
