import numpy as np

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    v = config["momentum"] * v - config["learning_rate"]*dw
    next_w = w + v
    config["velocity"] = v
    return next_w, config

def rmsprop(w, dw, config=None):
    if config == None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    keys = ["learning_rate","decay_rate","epsilon","cache"]
    lr, dr, eps, cache = (config.get(key) for key in keys)

    config["cache"] = dr * cache + (1 - dr) * dw**2
    next_w = w - lr * dw / (np.sqrt(config["cache"]) + eps)

    return next_w, config

def adam(w, dw, config=None):
    if config == None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    keys = ["learning_rate","beta1","beta2","epsilon","m","v","t"]
    lr, bt1, bt2, eps, m, v, t = (config.get(key) for key in keys)
    config["t"] = t = t + 1
    config["m"] = m = bt1 * m + (1 - bt1) * dw
    mt = m / (1 - bt1**t)
    config["v"] = v = bt2 * v + (1 - bt2) * (dw**2)
    vt = v / (1 - bt2**t)
    next_w = w - lr * mt / (np.sqrt(vt) + eps)

    return next_w, config