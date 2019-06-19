# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2019/6/18 20:15'
import numpy as np
import matplotlib.pyplot as plt

def dbmoon(N=100, d=2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)

        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    print(data)
    db_moon = data[0:N, :]
    print(db_moon)

    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon

if __name__ == '__main__':
    N = 100#点的个数
    d = -2#
    r = 10
    w = 2
    a = 0.1
    num_MSE = []
    num_step = []
    data = dbmoon(N, d, r, w)
    print('##############')
    #print(data)
    plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2 * N, 0], data[N:2 * N, 1], 'b*')
    #双月数据集，0-N个标红色
    #N-2N个标注蓝色
    #绘图后将其保存为训练测试集合
    plt.show()

    with open('data.txt','w+',encoding='utf-8') as f:
        for i in range(N):
            f.write(str(data[i,0])+','+str(data[i,1])+','+'0')
            f.write('\n')
        for i in range(N,2 * N):
            f.write(str(data[i,0])+','+str(data[i,1])+','+'1')
            f.write('\n')
    #plt.savefig('data.png')

