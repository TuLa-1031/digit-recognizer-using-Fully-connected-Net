from builtins import range
from builtins import object
import os
import numpy as np

from .layer_utils import *
from .layers import *

class FullyConnectedNet(object):
    def __init__(
        self,
        hidden_dims,
        input_dim=28*28,
        num_classes = 10,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
    ):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype =dtype
        self.params = {}

        for l, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f'W{l+1}'] = np.random.randn(i, j)*weight_scale
            self.params[f'b{l+1}'] = np.zeros(j)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        scores = None
        cache = {}
        for l in range(self.num_layers):
            keys = [f'W{l+1}', f'b{l+1}']
            w, b = (self.params.get(k, None) for k in keys)
            X, cache[l] = affine_relu_forward(X, w, b)

        scores = X

        if mode == "test":
            return scores
        
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum([np.sum(W**2) for k, W in self.params.items() if 'W' in k]))

        for l in reversed(range(self.num_layers)):
            dout, dW, db = affine_relu_backward(dout, cache[l])
            grads[f'W{l+1}'] = dW + self.reg*self.params[f'W{l+1}']
            grads[f'b{l+1}'] = db

        return loss, grads
    
    def save(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True