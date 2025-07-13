import numpy as np
from builtins import range

def affine_forward(x, w, b):
    out = None
    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped @ w + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    x_reshaped = x.reshape(x.shape[0], -1)
    dx = (dout @ w.T).reshape(x.shape[0], *x.shape[1:])
    dw = x_reshaped.T @ dout
    db = dout.sum(axis=0)
    return dx, dw, db

def relu_forward(x):
    out = None
    out = np.maximum(x, 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx

def softmax_loss(x, y):
    loss, dx = None, None
    N = len(y)
    p = np.exp(x - x.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[range(N), y]).sum()
    loss /= N
    p[range(N), y] -= 1
    dx = p / N
    return loss, dx