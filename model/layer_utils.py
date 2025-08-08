from .layers import *

def forward(x, w, b, 
            gamma=None, beta=None, bn_param=None, 
            dropout_param=None, 
            last=False):
    bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None
    out, fc_cache = affine_forward(x, w, b)
    if last != True:
        if bn_param is not None:
            if "mode" in bn_param:
                out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
            else:
                out, ln_cache = layernom_forward(out, gamma, beta, bn_param)
        out, relu_cache = relu_forward(out)
        if dropout_param is not None:
            out, dropout_cache = dropout_forward(out, dropout_param)
    cache = fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache
    return out, cache

def backward(dout, cache):
    dgamma, dbeta = None, None
    fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache = cache
    if dropout_cache is not None:
        dout = dropout_backward(dout, dropout_cache)
    if relu_cache is not None:
        dout = relu_backward(dout,relu_cache)
    if bn_cache is not None:
        dout, gamma, dbeta = batchnorm_backward(dout, bn_cache)
    elif ln_cache is not None:
        dout, dgamma, dbeta = layernorm_backward(dout, ln_cache)
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta
