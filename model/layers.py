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

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype = x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    
    out, cache = None, None
    if mode == "train":
        mu = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var+eps)
        x_hat = (x-mu)/std
        out = x_hat*gamma + beta
        shape = bn_param.get("shape", (N, D))
        axis = bn_param.get("axis", 0)
        cache = x, mu, var, std, gamma, beta, x_hat, shape, axis, eps
        if axis==0: # dùng để cho layer normalization
            running_mean = momentum * running_mean + (1-momentum)*mu
            running_var = momentum*running_var + (1-momentum)*var
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_hat * gamma + beta
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache

def batchnorm_backward(dout, cache):
    x, mu, var, std, gamma, beta, x_hat, shape, axix, eps = cache
    N, D = x.shape

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmu = np.sum(dxhat * -1 / std, axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    dx = dxhat / std + dvar * 2 * (x - mu) / N + dmu / N

    return dx, dgamma, dbeta

def layernom_forward(x, gamma, beta, ln_param):
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)

    bn_param = {"axis": 1, **ln_param}
    [gamma, beta] = np.atleast_2d(gamma, beta)
    out, cache = batchnorm_backward(x.T, gamma.T, beta.T, bn_param)
    return out, cache

def layernorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)
    dx = dx.T
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    mask, out = None, None
    if mode == "test":
        out = x
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx

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