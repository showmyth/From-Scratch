import numpy as np

class Layer_Norm:
    def __init__(self, d_model, eps = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, X):
        mean = np.mean(X, axis = -1, keepdims=True)
        var = np.var(X, axis = -1, keepdims=True)
        norm = (X - mean) / np.sqrt(var + self.eps)
        return (self.gamma * norm + self.beta) 


# TEST

if __name__ == "__main__":
    np.random.seed(67)

    x = np.random.randn(2, 5, 8)
    layer_norm = Layer_Norm(d_model = 8)
    out = layer_norm.forward(x)
    assert out.shape == x.shape
    print(f"original dims: {x.shape} , applying layer norm: {out.shape}")