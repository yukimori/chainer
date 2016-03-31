import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check


class TestCRF1d(unittest.TestCase):

    batch = 2

    def setUp(self):
        self.cost = numpy.random.uniform(-1, 1, (3, 3)).astype(numpy.float32)
        self.xs = [numpy.random.uniform(
            -1, 1, (self.batch, 3)).astype(numpy.float32) for _ in range(2)]
        self.ys = [numpy.random.randint(
            0, 3, (self.batch,)).astype(numpy.int32) for _ in range(2)]
        self.gy = numpy.random.uniform(
            -1, 1, (self.batch,)).astype(numpy.float32)

    def test_forward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.xs[i]) for i in range(2)]
        ys = [chainer.Variable(self.ys[i]) for i in range(2)]
        log_p = functions.crf1d(cost, xs, ys)

        z = numpy.zeros((self.batch,), numpy.float32)
        for y1 in range(3):
            for y2 in range(3):
                z += numpy.exp(self.xs[0][range(self.batch), y1] +
                               self.xs[1][range(self.batch), y2] +
                               self.cost[y1, y2])

        score = (self.xs[0][range(self.batch), self.ys[0]] +
                 self.xs[1][range(self.batch), self.ys[1]] +
                 self.cost[self.ys[0], self.ys[1]])

        expect = -(score - numpy.log(z))
        gradient_check.assert_allclose(log_p.data, expect)

    def test_backward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.xs[i]) for i in range(2)]
        ys = [chainer.Variable(self.ys[i]) for i in range(2)]
        log_p = functions.crf1d(cost, xs, ys)
        log_p.grad = self.gy
        log_p.backward()

    def test_viterbi(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.xs[i]) for i in range(2)]
        s, path = functions.loss.crf1d.crf1d_viterbi(cost, xs)

        best_paths = [numpy.empty((self.batch,), numpy.int32)
                      for i in range(len(xs))]
        for b in range(self.batch):
            best_path = None
            best_score = 0
            for y1 in range(3):
                for y2 in range(3):
                    score = self.xs[0][b, y1] + self.xs[1][b, y2] + self.cost[y1, y2]
                    if best_path is None or best_score < score:
                        best_path = [y1, y2]
                        best_score = score
            best_paths[0][b] = best_path[0]
            best_paths[1][b] = best_path[1]

        numpy.testing.assert_array_equal(path, best_paths)
