from builtins import range
from builtins import object
import os
import numpy as np

from .layer_utils import *
from .layers import *

class FCNet(object):
    def __init__(
        self,
        hidden_dims,
        input_dim=28*28,
        num_classes = 10,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        normalization=None,
        dropout=1,
    ):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype =dtype
        self.params = {}
        self.normalization=normalization
        self.use_dropout = dropout!=1

        for l, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f'W{l+1}'] = np.random.randn(i, j)*weight_scale
            self.params[f'b{l+1}'] = np.zeros(j)
            if self.normalization and l < self.num_layers-1:
                self.params[f'gamma{l+1}'] = np.ones(j)
                self.params[f'beta{l+1}'] = np.zeros(j)
        self.dropout_param={}
        if self.use_dropout:
            self.dropout_param={"mode": "train", "p": dropout}
        self.bn_param={}
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers-1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers-1)]
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set mode cho dropout và batchnorm
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        caches = []
        out = X

        # ----- Forward -----
        for i in range(self.num_layers - 1):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']

            out, fc_cache = affine_forward(out, W, b)

            if self.normalization == 'batchnorm':
                gamma = self.params[f'gamma{i+1}']
                beta = self.params[f'beta{i+1}']
                bn_param = self.bn_params[i]
                out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
            elif self.normalization == 'layernorm':
                gamma = self.params[f'gamma{i+1}']
                beta = self.params[f'beta{i+1}']
                ln_param = {}
                out, bn_cache = layernorm_forward(out, gamma, beta, ln_param)
            else:
                bn_cache = None

            out, relu_cache = relu_forward(out)

            if self.use_dropout:
                out, do_cache = dropout_forward(out, self.dropout_param)
            else:
                do_cache = None

            caches.append((fc_cache, bn_cache, relu_cache, do_cache))

        # Last affine layer (no ReLU, no dropout, no norm)
        W = self.params[f'W{self.num_layers}']
        b = self.params[f'b{self.num_layers}']
        scores, fc_cache = affine_forward(out, W, b)
        caches.append((fc_cache, None, None, None))

        if mode == 'test':
            return scores

        # ----- Loss -----
        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            W = self.params[f'W{i+1}']
            loss += 0.5 * self.reg * np.sum(W * W)

        # ----- Backward -----
        grads = {}

        # Last layer backward
        dout, dW, db = affine_backward(dscores, caches[-1][0])
        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        dout_prev = dout

        for i in reversed(range(self.num_layers - 1)):
            fc_cache, bn_cache, relu_cache, do_cache = caches[i]

            if self.use_dropout:
                dout_prev = dropout_backward(dout_prev, do_cache)

            dout_prev = relu_backward(dout_prev, relu_cache)

            if self.normalization == 'batchnorm':
                dout_prev, dgamma, dbeta = batchnorm_backward(dout_prev, bn_cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
            elif self.normalization == 'layernorm':
                dout_prev, dgamma, dbeta = layernorm_backward(dout_prev, bn_cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta

            dout_prev, dW, db = affine_backward(dout_prev, fc_cache)
            grads[f'W{i+1}'] = dW + self.reg * self.params[f'W{i+1}']
            grads[f'b{i+1}'] = db

        return loss, grads
