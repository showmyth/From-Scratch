import numpy as np
try:
    from .layers.linear import Linear
    from .layers.activations import relu, gelu
except ImportError:
    from layers.linear import Linear
    from layers.activations import relu, gelu

class Feed_Forward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = gelu
        self.linear1 = Linear(self.d_model, self.d_ff)
        self.linear2 = Linear(self.d_ff, self.d_model)

    def forward(self, X):
        # pass through layer 1 => linear1
        out = self.linear1.forward(X)
        # pass through activation
        out = self.activation(out)
        # pass through layer 2 => linear2
        out = self.linear2.forward(out)
        return out

if __name__ == "__main__":
    B, S, d_model, d_ff = 2 ,5 ,8, 16
    x = np.random.randn(B, S, d_model)
    ff = Feed_Forward(d_model, d_ff)
    out = ff.forward(x)
    assert out.shape == x.shape
    print(f"{x.shape} -> {out.shape} : feed forward test passed!")