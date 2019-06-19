# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2019/6/18 20:33'
# -*- coding: utf-8 -*-
import math
import random
import string
import pickle

flowerLables = {0:'红色',
                 1:'蓝色'}
random.seed(0)
# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b-a)*random.random() + a

# 生成大小 I*J 的矩阵，默认零矩阵 (当然，亦可用 NumPy 提速)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# 函数 sigmoid，这里采用 tanh，因为看起来要比标准的 1/(1+e^-x) 漂亮些
def sigmoid(x):
    return math.tanh(x)

# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    ''' 三层反向传播神经网络 '''
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1 # 增加一个偏差节点
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # 最后建立动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        # print self.ao
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        ''' 反向传播 '''
        # if len(targets) != self.no:
        #     raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        # for k in range(len(targets)):
        #     error = error + 0.5*(targets[k]-self.ao[k])**2
        error += 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        count = 0
        for p in patterns:
            target = flowerLables[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            print(p[0], ':', target, '->', flowerLables[index])
            count += (target == flowerLables[index])
            result_str = flowerLables[index]
            if target == result_str:
                count += 1
            else:
                pass
        accuracy = float(count/len(patterns))
        print('accuracy: %-.9f' % accuracy)

    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.1, M=0.01):
        # N: 学习速率(learning rate)
        # M: 动量因子(momentum factor)
        train_line = []
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('误差 %-.9f' % error)
                train_line.append(error)
        return train_line

import numpy as np
import pandas as pd

def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]

def bp_net():
    data = []
    # read dataset
    raw = pd.read_csv('data.txt')
    raw_data = raw.values
    raw_feature = raw_data[0:,0:2]

    ###############数据归一化操作

    from sklearn.preprocessing import StandardScaler

    # 标准化，返回值为标准化后的数据
    new_feature=StandardScaler().fit_transform(raw_feature)

    print(raw_feature)
    #raw_feature=Normalize(raw_feature)
    raw_feature=new_feature
    #######################数据归一化完成



    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][2] == '优':
           ele.append([1,0])#,0
        elif raw_data[i][2] == '良':
            ele.append([0,1])#,0
        data.append(ele)

    # print data

    # 随机排列data
    random.shuffle(data)

    print(len(data))

    training = data[0:150]
    test = data[151:]
    #训练集跟测试机的划分

    #筛选奇异值
    index=0
    new_training=[]
    for p in training:
        try:
            inputs = p[0]
            targets = p[1]
            new_training.append(p)
        except:
            pass

    nn = NN(2,9,1)
    #隐含层节点个数
    line=nn.train(new_training,iterations=1000)

    # save weights
    with open('wi.txt', 'wb') as wif:
       pickle.dump(nn.wi, wif)
    with open('wo.txt', 'wb') as wof:
       pickle.dump(nn.wo, wof)

    nn.test(test)

    import matplotlib.pyplot as plt
    plt.plot(line, label='loss')
    print(line)
    plt.legend()
    plt.show()
    print('ok')

if __name__ == '__main__':

    bp_net()

