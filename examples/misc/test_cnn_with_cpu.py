import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

"""
http://qiita.com/HirofumiYashima/items/417a83db10e464f56797
"""

# Network definition
class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, 32, 5),
            conv2 = L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024,10),
            )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)


