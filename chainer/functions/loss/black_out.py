from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import reshape
from chainer.functions.connection import embed_id
from chainer.functions.math import exponential
from chainer.functions.math import logsumexp
from chainer.functions.math import matmul
from chainer.functions.math import sum as _sum


def black_out(x, t, W, samples):
    batch_size = x.data.shape[0]

    neg_emb = embed_id.embed_id(samples, W)
    neg_y = matmul.batch_matmul(neg_emb, x)
    neg_y = reshape.reshape(neg_y, neg_y.data.shape[:-1])

    pos_emb = expand_dims.expand_dims(embed_id.embed_id(t, W), 1)
    pos_y = matmul.batch_matmul(pos_emb, x)
    pos_y = reshape.reshape(pos_y, pos_y.data.shape[:-1])

    logz = logsumexp.logsumexp(concat.concat([pos_y, neg_y]), axis=1)
    blogz, bneg_y = broadcast.broadcast(
        reshape.reshape(logz, (batch_size, 1)), neg_y)
    ny = exponential.log(1 - exponential.exp(bneg_y - blogz))
    py = reshape.reshape(pos_y, (batch_size,))
    loss = py - logz + _sum.sum(ny, axis=1)
    return -_sum.sum(loss) / batch_size
