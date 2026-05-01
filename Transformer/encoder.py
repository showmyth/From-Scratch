import numpy as np

try:
    from .attention import MultiHead_Attention
    from .ff import Feed_Forward
    from .layers.layer_norm import Layer_Norm
except ImportError:
    from attention import MultiHead_Attention
    from ff import Feed_Forward
    from layers.layer_norm import Layer_Norm


class Encoder_Block:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.self_attention = MultiHead_Attention(d_model, num_heads)
        self.norm1 = Layer_Norm(d_model)
        self.feed_forward = Feed_Forward(d_model, d_ff)
        self.norm2 = Layer_Norm(d_model)

    def forward(self, X, mask=None):
        self.X = X
        # Run multi-head self-attention on X
        out = self.self_attention.forward(X, mask)
        # Add the residual connection
        residual1 = X + out
        # Apply the first layer norm
        out_temp = self.norm1.forward(residual1)
        # Run the feed-forward block
        ff1 = self.feed_forward.forward(out_temp)
        # Add the second residual connection
        residual2 = residual1 + ff1
        # Apply the second layer norm
        out = self.norm2.forward(residual2)
        return out


class Encoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.layers = [
            Encoder_Block(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def forward(self, X, mask=None):
        out = X
        for layer in self.layers:
            out = layer.forward(out, mask)
        return out


if __name__ == "__main__":
    np.random.seed(67)

    B, S, d_model, num_heads, d_ff, num_layers = 2, 5, 8, 2, 16, 2
    X = np.random.randn(B, S, d_model)

    encoder = Encoder(d_model, num_heads, d_ff, num_layers)

    # Replace this with a real forward-pass test after you implement the logic.
    out = encoder.forward(X)
    assert out.shape == X.shape
    print(f"{X.shape} -> {out.shape} : Encoder test passed!")
