from chainer import cuda
from chainer.fuctions import minmax
from chainer import variable


def hard_sigmoid(x):
    xp = cuda.get_array_module(x)
    zero = variable.Variable(xp.zeros_like(x.data))
    one = variable.Variable(xp.ones_like(x.data))
    return minmax.max(zero, minmax.min(one, x * 0.2 + 0.5))
