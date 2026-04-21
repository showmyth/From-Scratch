import numpy as np


def softmax(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    X    : ndarray, any shape
    axis : axis along which to compute softmax, default -1 (last dim)
           in attention this is the score/key dimension

    Return -> an N-D array, same shape as X, values in (0, 1) summing to 1 along `axis`
    """
    # subtract max along last axis (keepdims=True) for stability
    X_max = np.max(X, axis = -1, keepdims=True)
    X = X - X_max
    # exponentiate
    exp_X = np.exp(X)
    # divide by sum along same axis (keepdims=True)
    sm_X = exp_X / np.sum(exp_X, axis = -1, keepdims=True)
    return sm_X


def relu(X: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: max(0, x)
    """
    relu_x = np.maximum(0, X)
    return relu_x


def gelu(X: np.ndarray) -> np.ndarray:
    """
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    gelu_x = 0.5 * X * (1 + np.tanh(np.sqrt(2/np.pi) * (X + 0.044715 * X**3)))
    return gelu_x


# TEST

if __name__ == "__main__":
    np.random.seed(67)

    # softmax 1 -> outputs sum to 1
    x = np.random.randn(2, 5, 8)
    out = softmax(x)
    assert out.shape == x.shape
    assert np.allclose(out.sum(axis=-1), 1.0)
    print(f"softmax : {x.shape} -> {out.shape}  ✓")

    # softmax 2 -> no overflow
    x_large = np.array([1000.0, 1001.0, 1002.0])
    out_large = softmax(x_large)
    assert not np.any(np.isnan(out_large)), "overflow — stability fix missing"
    print(f"softmax stability : {out_large}  ✓")

    # relu 1 -> -ve == 0 , +ve remain unchanged
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = relu(x)
    assert np.all(out >= 0)
    print(f"relu    : {x} -> {out}  ✓")

    # gelu 1 -> smooth out values
    out = gelu(x)
    assert out.shape == x.shape
    print(f"gelu: {x} -> {out}✓")