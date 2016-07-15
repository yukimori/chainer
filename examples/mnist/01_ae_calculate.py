#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda

from pprint import pprint
import matplotlib.pyplot as plt
import sys, time, math

import chainer.links as L
from chainer import optimizers
from chainer import serializers

from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F

import data
import net

parser = argparse.ArgumentParser(description='AutoEncoder example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=996, type=int,
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
# ノイズ付加有無
noised = False

print('# unit: {}'.format(n_units))
print('# Minibatch-size: {}'.format(batchsize))
print('# epoch: {}'.format(n_epoch))
#print('Network type: {}'.format())
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
#y_train, y_test = np.split(mnist['target'], [N])
y_train, y_test = np.split(mnist['data'].copy(), [N])
print(type(x_train))
print("x_train.shape:{0},x_test.shape{1}".format(x_train.shape, x_test.shape))
print("y_train.shape:{0},y_test.shape{1}".format(y_train.shape, y_test.shape))
print("x_train[0]:{}".format(x_train[0]))

N_test = y_test.shape[0]
print("N_test : {}".format(N_test))

if noised:
    # Add noise
    noise_ratio = 0.2
    for data in mnist.data:
        perm = np.random.permutation(mnist.data.shape[1])[:int(mnist.data.shape[1]*noise_ratio)]
        data[perm] = 0.0

# AutoEncoderのモデルの設定
# 入力 784次元、出力 784次元, 2層
model = FunctionSet(l1=F.Linear(784, n_units),
                    l2=F.Linear(n_units, 784))

print("created model.")

# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    y = F.dropout(F.relu(model.l1(x)),  train=train)
    x_hat  = F.dropout(model.l2(y),  train=train)
    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

print("setup optimizer.")

l1_W = []
l2_W = []
l1_b = []
l2_b = []

train_loss = []
test_loss = []
test_mean_loss = []

prev_loss = -1
loss_std = 0

loss_rate = []

# Learning loop
for epoch in xrange(1, n_epoch+1):
    print('# epoch: {}'.format(epoch))
    start_time = time.clock()

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
#        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_loss += float(loss.data) * batchsize

#    print '\ttrain mean loss={} '.format(sum_loss / N)
    print('# \ttrain mean loss={}'.format(sum_loss / N))
    
    # evaluation
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        loss = forward(x_batch, y_batch, train=False)
#        print('{0} : {1}'.format(i,loss.data))
        
        test_loss.append(loss.data)
#        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_loss += float(loss.data) * batchsize

    loss_val = sum_loss / N_test
    print('sum_loss={}'.format(sum_loss))
    print('N_test={}'.format(N_test))
#    print '\ttest  mean loss={}'.format(loss_val)
    print('# \ttest mean loss={}'.format(loss_val))
    if epoch == 1:
        loss_std = loss_val
        loss_rate.append(100)
    else:
#        print '\tratio :%.3f'%(loss_val/loss_std * 100)
        print('# \tratio : %.3f' % (loss_val / loss_std * 100))
        loss_rate.append(loss_val/loss_std * 100)
        
    if prev_loss >= 0:
        diff = loss_val - prev_loss
        ratio = diff/prev_loss * 100
#        print '\timpr rate:%.3f'%(-ratio)
        print('# \timpr rate : %.3f' % (-ratio))

    prev_loss = sum_loss / N_test
    test_mean_loss.append(loss_val)
    print("  \ttest_mean_loss:{}".format(test_mean_loss))
    
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    end_time = time.clock()
#    print "\ttime = %.3f" %(end_time-start_time)
    print('# \ttime : %.3f' % (end_time - start_time))

print("finished model.")

# Save the model and the optimizer
print('save the model')
serializers.save_npz('ae.model', model)
print('save the optimizer')
serializers.save_npz('ae.state', optimizer)

plt.style.use('ggplot')
# draw a image of handwriting number
def draw_digit_ae(data, cnt, length, _type):
    size = 28
    coln = min(math.ceil(math.sqrt(length)), 10)
    plt.subplot(math.ceil(length/float(coln)), int(coln), cnt)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,28)
    plt.ylim(0,28)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

# Draw mean loss graph
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss,"o-",lw=1)
plt.title("")
plt.ylabel("mean loss")
#plt.show()
plt.xlabel("epoch")
plt.savefig("loss_graph.png")

# 入力と出力を可視化
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,25))

num = 100
cnt = 0
ans_list  = []
pred_list = []
for idx in np.random.permutation(N_test)[:num]:
    xxx = x_test[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(Variable(xxx.reshape(1,784)))),  train=False)
    y = model.l2(h1)
    ans_list.append(x_test[idx])
    pred_list.append(y)

print("start input/output.gif")
col_size=10
data_size = len(ans_list + pred_list)
for i in six.moves.range(0, len(ans_list), col_size):
    print("start:{}".format(i+1))
    for j, (ans_elem, pred_elem) in enumerate(zip(ans_list[i:i+col_size],pred_list[i:i+col_size])):
        ans_pos = (2*i) + j + 1
        pred_pos = ans_pos + col_size
        draw_digit_ae(ans_elem, ans_pos, data_size, "ans")
        draw_digit_ae(pred_elem.data, pred_pos, data_size, "pred")

# for i in range(int(num/10)):
#     for j in range (10):
#         img_no = i*10+j
#         pos = (2*i)*10+j
#         draw_digit_ae(ans_list[img_no],  pos+1, 20, 10, "ans")

# for i, pred_elem in enumerate(pred_list):
#     draw_digit_ae(pred_elem.data, i+1, len(pred_list), "pred")
        
    # for j in range (10):
    #     img_no = i*10+j
    #     pos = (2*i+1)*10+j
    #     draw_digit_ae(pred_list[i*10+j].data, pos+1, 20, 10, "pred")
plt.savefig("inputoutput.png")

# W(1)を可視化
plt.style.use('fivethirtyeight')
# draw digit images
def draw_digit_w1(data, cnt, length):
    size = 28
    coln = min(math.ceil(math.sqrt(length)), 15)
    plt.subplot(math.ceil(length/float(coln)), int(coln), cnt)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")


plt.figure(figsize=(15,70))
# l1_Wにはepochごとの重みが保存されていると思われる
w1_lastlayer = l1_W[-1].data # wa_lastlayer:[n_unit][784]
for i,w1_elem in enumerate(w1_lastlayer):
#    print("w1_elem :{}".format(type(w1_elem)))
#    print("w1_elem :{}".format(w1_elem.shape))
    if (i % 100) == 0:
        print("{} done.".format(i+1), end=" ")
#    draw_digit_w1(l1_W[9][i], cnt, i, len(l1_W[9][i]))
    draw_digit_w1(w1_elem, (i+1), w1_lastlayer.shape[0])
plt.savefig("w1.png")
print("")

# W(2).Tを可視化
plt.style.use('fivethirtyeight')
# draw digit images
def draw_digit2(data, cnt, length):
    size = 28
    coln = min(math.ceil(math.sqrt(length)), 15)
    plt.subplot(math.ceil(length/float(coln)), int(coln), cnt)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

w2_T = np.array(l2_W[-1].data).T
plt.figure(figsize=(15,30))
for i,w2_T_elem in enumerate(w2_T):
    draw_digit2(w2_T_elem, (i+1), w2_T.shape[0])
plt.savefig("w2.png")

# # model作成
# # Prepare multi-layer perceptron model, defined in net.py
# if args.net == 'simple':
#     # net.pyに実装が移動しているが作成されているのは3層NN
#     # 入力データが784次元なので入力素子は784個。中間層はn_unitsでデフォルト1000と指定
#     # 出力は数字を識別するので10個
#     # L:chainer.links MLP 各層が100個のユニットをもつ三層のネットワーク
#     # Classifierクラスは損失と精度を計算し、損失値を返却する
#     model = L.Classifier(net.MnistMLP(784, n_units, 10))
#     if args.gpu >= 0:
#         cuda.get_device(args.gpu).use()
#         model.to_gpu()
#     xp = np if args.gpu < 0 else cuda.cupy
# elif args.net == 'parallel':
#     cuda.check_cuda_available()
#     model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
#     xp = cuda.cupy

# # Setup optimizer
# # optimizerにmodelを設定する
# # 最適化手法としてAdamを利用
# optimizer = optimizers.Adam()
# optimizer.setup(model)

# # Init/Resume
# if args.initmodel:
#     print('Load model from', args.initmodel)
#     serializers.load_npz(args.initmodel, model)
# if args.resume:
#     print('Load optimizer state from', args.resume)
#     serializers.load_npz(args.resume, optimizer)

# # Learning loop
# # sixはバージョンニュートラルなコードを記述するためのライブラリ
# # six.rangeはpython2ではxrangeとして処理することができる（python3ではrange）
# for epoch in six.moves.range(1, n_epoch + 1):
#     print('epoch', epoch)

#     # training
#     # permutationにより0からN-1までの数値をランダムに並び替えたarrayを返却する
#     perm = np.random.permutation(N)
#     sum_accuracy = 0
#     sum_loss = 0
#     start = time.time()

#     # batchsizeのデフォルト:100 Nはデフォルト60000
#     for i in six.moves.range(0, N, batchsize):
#         # 配列からChainerのVariableという型（クラス）のオブジェクトに変換するという部分？
#         # trainデータからbatchsize分のデータを切り出している
#         # xpにはgpu=-1のときはnpが入っているのでnumpyで計算している
#         # permを使うことで{x,y}_trainからランダムにデータを取得している（かつforで回せる）
#         x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
#         t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

#         # Pass the loss function (Classifier defines it) and its arguments
#         # xは手書きデータ、tが正解データ batchsize分のデータを投入している
#         # ここでlossなども計算されて更新されている模様
#         # modelを損失関数としてoptimizerに渡している
#         optimizer.update(model, x, t)
#         # 以下の3行と同等になる。ネットワークがforward計算によって定義され誤差逆伝搬法により更新される
#         # model.zerograds()
#         # loss = model(x, t)
#         # loss.backward()
#         # optimizer.update()

#         # modelの損失を取得して保存しているか？
#         if epoch == 1 and i == 0:
#             with open('graph.dot', 'w') as o:
#                 g = computational_graph.build_computational_graph(
#                     (model.loss, ))
#                 o.write(g.dump())
#             print('graph generated')

#         # 合計lossを計算
#         sum_loss += float(model.loss.data) * len(t.data)
#         sum_accuracy += float(model.accuracy.data) * len(t.data)
#     end = time.time()
#     elapsed_time = end - start
#     throughput = N / elapsed_time
#     print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
#         sum_loss / N, sum_accuracy / N, throughput))

#     # evaluation
#     sum_accuracy = 0
#     sum_loss = 0
#     for i in six.moves.range(0, N_test, batchsize):
#         x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
#                              volatile='on')
#         t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
#                              volatile='on')
#         # この呼び方は__call__メソッドを呼び出しているのか？->多分YES
#         loss = model(x, t)
#         sum_loss += float(loss.data) * len(t.data)
#         sum_accuracy += float(model.accuracy.data) * len(t.data)

#     print('test  mean loss={}, accuracy={}'.format(
#         sum_loss / N_test, sum_accuracy / N_test))

