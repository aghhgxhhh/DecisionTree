import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# 计算两个向量之间的欧式距离
def calDist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 获取一个点的ε-邻域
def getNeibor(p, datasets, e):
    neighbors = []
    for i, x in enumerate(datasets):
        if calDist(p, x) <= e:
            neighbors.append(i)
    return neighbors

# 密度聚类算法
def DBSCAN(datasets, e=0.5, minPts=5):
    '''
    :param datasets: array[n, features_num]
    :param e: float 判定相邻的最大距离
    :param minPts: int 判定为核心节点的邻居节点数目
    :return: classes: dict 聚类结果
    '''
    coreObjs = [] # 核心节点集合
    neighbors = [] # 存储邻居节点
    # 存储邻居节点信息，并找出所有核心节点
    for i, x in enumerate(datasets):
        neighbors.append(getNeibor(x, datasets, e))
        if len(neighbors[i]) >= minPts:
            coreObjs.append(i)

    k = 0  # 初始化聚类簇
    visited = list(0 for _ in range(len(datasets))) # 记录样本点有没有被访问
    classes = {} # 记录类别信息

    while len(coreObjs) > 0:
        classes[k] = []
        # 随机选取一个核心对象
        i = random.choice(coreObjs)
        que = deque()
        que.append(i)
        while len(que) > 0:
            q = que.popleft()
            coreObjs.remove(q)
            for neigbor in neighbors[q]:
                if visited[neigbor]:
                    continue
                visited[neigbor] = 1
                classes[k].append(neigbor)
                if neigbor in coreObjs:  # 将核心对象入队
                    que.append(neigbor)
        k += 1
    return classes


def plot(datasets, classes):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for k, indexes in classes.items():
        plt.scatter(datasets[indexes][:, 0], datasets[indexes][:, 1], marker='o', color=color[k], label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    datasets = np.loadtxt('./datasets.txt', dtype=float, delimiter=',')  # 读入csv文件
    classes = DBSCAN(datasets, 0.11, 5)
    plot(datasets, classes)
