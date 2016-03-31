import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check


class TestCRF1d(unittest.TestCase):
    batch = 2

    def setUp(self):
        self.cost = numpy.random.uniform(-1, 1, (3, 3)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (self.batch, 3)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, (self.batch, 3)).astype(numpy.float32)
        self.y1 = numpy.random.randint(0, 3, (self.batch,)).astype(numpy.int32)
        self.y2 = numpy.random.randint(0, 3, (self.batch,)).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, (self.batch,)).astype(numpy.float32)

    def test_forward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.x1), chainer.Variable(self.x2)]
        ys = [chainer.Variable(self.y1), chainer.Variable(self.y2)]
        log_p = functions.crf1d(cost, xs, ys)

        z = numpy.zeros((self.batch,), numpy.float32)
        for y1 in range(3):
            for y2 in range(3):
                z += numpy.exp(self.x1[range(self.batch), y1] +
                               self.x2[range(self.batch), y2] +
                               self.cost[y1, y2])

        score = (self.x1[range(self.batch), self.y1] +
                 self.x2[range(self.batch), self.y2] +
                 self.cost[self.y1, self.y2])

        expect = -(score - numpy.log(z))
        gradient_check.assert_allclose(log_p.data, expect)

    def test_backward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.x1), chainer.Variable(self.x2)]
        ys = [chainer.Variable(self.y1), chainer.Variable(self.y2)]
        log_p = functions.crf1d(cost, xs, ys)
        log_p.grad = self.gy
        log_p.backward()
