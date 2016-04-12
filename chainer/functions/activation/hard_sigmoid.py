from chainer.functions.math import clip


def hard_sigmoid(x):
    return clip.clip(x * 0.2 + 0.5, 0.0, 1.0)
