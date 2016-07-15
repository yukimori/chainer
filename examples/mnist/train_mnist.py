#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import argparse
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

# 確率的勾配降下法で学習させる際の1回分のバッチサイズ
batchsize = args.batchsize
# 学習の繰り返し回数
n_epoch = args.epoch
# 中間層の数
n_units = args.unit

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('')

# Prepare dataset
print('load MNIST dataset')
# データの取得 scikit-learn依存から独自に変更されている？
mnist = data.load_mnist_data()
# mnistデータは70000件の784次元ベクトルデータと思われる
mnist['data'] = mnist['data'].astype(np.float32)
# 0-1のデータに変換
mnist['data'] /= 255
# 正解（教師）のデータ（dataがどの数値かを表す）
mnist['target'] = mnist['target'].astype(np.int32)

# 学習データをN（60000）、検証データを残りの個数と設定
# dataが画像データ、targetは画像がどの数値を表す数値データ
N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
print(type(x_train))
print("x_train.shape:{0},x_test.shape{1}".format(x_train.shape, x_test.shape))
print("y_train.shape:{0},y_test.shape{1}".format(y_train.shape, y_test.shape))

N_test = y_test.size


# model作成
# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    # net.pyに実装が移動しているが作成されているのは3層NN
    # 入力データが784次元なので入力素子は784個。中間層はn_unitsでデフォルト1000と指定
    # 出力は数字を識別するので10個
    # L:chainer.links MLP 各層が100個のユニットをもつ三層のネットワーク
    # Classifierクラスは損失と精度を計算し、損失値を返却する
    model = L.Classifier(net.MnistMLP(784, n_units, 10))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
    xp = cuda.cupy

# Setup optimizer
# optimizerにmodelを設定する
# 最適化手法としてAdamを利用
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Learning loop
# sixはバージョンニュートラルなコードを記述するためのライブラリ
# six.rangeはpython2ではxrangeとして処理することができる（python3ではrange）
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    # permutationにより0からN-1までの数値をランダムに並び替えたarrayを返却する
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()

    # batchsizeのデフォルト:100 Nはデフォルト60000
    for i in six.moves.range(0, N, batchsize):
        # 配列からChainerのVariableという型（クラス）のオブジェクトに変換するという部分？
        # trainデータからbatchsize分のデータを切り出している
        # xpにはgpu=-1のときはnpが入っているのでnumpyで計算している
        # permを使うことで{x,y}_trainからランダムにデータを取得している（かつforで回せる）
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        # xは手書きデータ、tが正解データ batchsize分のデータを投入している
        # ここでlossなども計算されて更新されている模様
        # modelを損失関数としてoptimizerに渡している
        optimizer.update(model, x, t)
        # 以下の3行と同等になる。ネットワークがforward計算によって定義され誤差逆伝搬法により更新される
        # model.zerograds()
        # loss = model(x, t)
        # loss.backward()
        # optimizer.update()

        # modelの損失を取得して保存しているか？
        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ))
                o.write(g.dump())
            print('graph generated')

        # 合計lossを計算
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        # この呼び方は__call__メソッドを呼び出しているのか？->多分YES
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
