# -*- coding:utf-8 -*-

from __future__ import print_function
from sklearn.datasets import load_iris
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets
from chainer.training import extensions

iris = load_iris()
X = iris.data

# print(X)
# print("shape:", X.shape)

X = X.astype(np.float32)
# print(X)

Y = iris.target
# flatten:ネストしたリストをフラットにする
Y = Y.flatten().astype(np.int32)
# print(Y)
# print(Y.shape)

train, test = datasets.split_dataset_random(chainer.datasets.TupleDataset(X,Y), 100)
# print("train:", train)
# print("test:", test)

# イテレータの設定
# http://docs.chainer.org/en/stable/reference/iterators.html
# 10 - 1回の学習でのバッチサイズ
# train - 学習データ
train_iter = chainer.iterators.SerialIterator(train, 10)
# print("train_iter:", train_iter)
test_iter = chainer.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

class IrisModel(chainer.Chain):
    def __init__(self):
        super(IrisModel, self).__init__(
            l1 = L.Linear(4, 100),
            l2 = L.Linear(100, 100),
            l3 = L.Linear(100, 3))

    # reluは活性化関数
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# モデルの作成
model = L.Classifier(IrisModel())

# 最適化関数の作成
optimizer = chainer.optimizers.Adam()
# モデル（ネットワーク）をセット
optimizer.setup(model)

"""学習処理"""
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
# Trainer:学習ループを抽象化する
trainer = training.Trainer(updater, (20, 'epoch'), out="result")
# 一定の期間で評価する（validation）
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
# ログとして出力する
trainer.extend(extensions.LogReport())
# printを使って現状をレポートする
otrainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
# プログレスバーを表示する
trainer.extend(extensions.ProgressBar())

trainer.run()

