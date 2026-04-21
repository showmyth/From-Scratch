# Z = X @ W + b

import numpy as np

class Linear:
    
    def __init__(self, inputs, outputs, bias: bool = True):
        self.inputs = inputs
        self.outputs = outputs

        # Xavier Weight Initilization
        scale = np.sqrt(2.0 / (inputs + outputs))
        self.W = np.random.randn(inputs, outputs) * scale # -> makes sure weights arent too large to cause exploding grads
        self.b = np.zeros(outputs)

        # Gradients to be used in backprop()
        self.dW = None
        self.db = None
        self.input_cache = None

# Call

    def call_forward(self, X):
        return self.forward(X)

# FOWARD PASS

    def forward(self, X: np.ndarray) -> np.ndarray :
        """ 
        given X of shape (..., inputs)

        Compute :
        Z <- W @ X + b

        to maintain W: 
        we use X @ W + b X @ W => (, inputs) @ (inputs , ouputs) === (, outputs)
        """
        self.input_cache = X # keep this to retain inputs for backprop computation

        if self.b is not None:
            Z = X @ self.W + self.b
        else:
            Z = X @ self.W 
        return Z


# BACKPROPAGATION (BACKWARD PASS)
#     def backprop(self, dA: np.ndarray) -> np.ndarray :
#         """ 
#         dA is the gradient that is used to compute:
#         dW <- X^T @ dA
#         db = sum over all dims
#         dX <- dA @ W^T
#         """

#         # flatten dimensions
#         shape = self.input_cache.shape # gives (Batch , Seq, inputs)
#         X_flatten = self.input_cache.reshape(-1, self.inputs) # (Batch * Seq, inputs)
#         dA_flatten = dA.reshape(-1, self.outputs) # (Batch * Seq, outputs)
        
#         # compute grads
#         self.dW = X_flatten.T @ dA_flatten # (inputs, B*S) @ (B*S, outputs) -> (inputs, outputs)
#         if self.b is not None:
#             self.db = dA_flatten.sum(axis = 0) # (out, 1)
#         dX = dA @ self.W.T
#         return dX

# # HELPER FUNCTIONS
#     def params(self) -> dict:
#         """Return parameters and their gradients for an optimiser loop."""
#         param_dict = {"W": (self.W, self.dW)}
#         if self.b is not None:
#             param_dict["b"] = (self.b, self.db)
#         return param_dict
 
#     def represent(self):
#         return (f"Linear(in={self.in_features}, "
#                 f"out={self.out_features}, "
#                 f"bias={self.b is not None})")


# TEST

if __name__ == "__main__":
    np.random.seed(67)
 
    B, S, D_in, D_out = 2, 5, 8, 16
 
    layer = Linear(D_in, D_out)
    print(layer)
 
    X    = np.random.randn(B, S, D_in)
    out  = layer.forward(X)
    assert out.shape == (B, S, D_out), f"forward shape mismatch: {out.shape}"
    print(f"Weights : {layer.W.flat[:10]} and so on , Bias: {layer.b}")
    print(f"forward  : {X.shape} -> {out.shape}")
 
#     dout = np.ones((B, S, D_out))
#     dX   = layer.backprop(dout)
#     assert dX.shape  == (B, S, D_in),        f"dX shape mismatch:  {dX.shape}"
#     assert layer.dW.shape == (D_in, D_out),  f"dW shape mismatch:  {layer.dW.shape}"
#     assert layer.db.shape == (D_out,),       f"db shape mismatch:  {layer.db.shape}"
#     print(f"backward : dX {dX.shape}, dW {layer.dW.shape}, db {layer.db.shape}")





