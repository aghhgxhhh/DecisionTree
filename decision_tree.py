from collections import Counter
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn import datasets

def calc_entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * np.log(p)
    return res

class Node:
    # 非叶子节点属性, depth, d(划分维度), value(划分值)
    # 叶子节点属性, depth, classes(预测类别)
    def __init__(self, depth):
        self.d = None
        self.value = None
        self.classes = None
        self.depth = depth
        self.lchild = None
        self.rchild = None

class DecisionTree:

    def __init__(self, max_depth=2):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._fit(X, y, 0)

    def _fit(self, X, y, depth):
        if depth > self.max_depth:
            return None
        node = Node(depth)
        entropy = calc_entropy(y)
        if entropy > 0 and depth < self.max_depth:  # 类别不一致或小于最大深度才进行划分
            best_d, best_v = self.try_split(X, y)
            node.d = best_d
            node.value = best_v
            X_left, X_right, y_left, y_right = self.split(X, y, best_d, best_v)
            node.lchild = self._fit(X_left, y_left, node.depth+1)
            node.rchild = self._fit(X_right, y_right, node.depth+1)
        if node.lchild == None and node.rchild == None:
            node.classes = Counter(y).most_common(1)[0][0]  # 叶子节点中最常见的类作为类别值
        return node

    def predict(self, X):
        predict_y = np.empty(len(X))
        for i, x in enumerate(X):
            node = self.root
            while node.lchild != None or node.rchild != None:
                if x[node.d] <= node.value:
                    node = node.lchild
                else:
                    node = node.rchild
            predict_y[i] = node.classes
        return predict_y

    # 分割
    def split(self, X, y, d, value):
        index_a = (X[:, d] <= value)
        index_b = (X[:, d] > value)
        return X[index_a], X[index_b], y[index_a], y[index_b]

    # 寻找最佳的划分维度和数值
    def try_split(self, X, y):
        best_entropy = float('inf')
        best_d, best_v = -1, -1
        for d in range(X.shape[1]):
            sorted_index = np.argsort(X[:, d])
            for i in range(1, len(X)):
                if X[sorted_index[i], d] != X[sorted_index[i - 1], d]:
                    v = (X[sorted_index[i], d] + X[sorted_index[i - 1], d]) / 2
                    X_l, X_r, y_l, y_r = self.split(X, y, d, v)
                    p_l, p_r = len(X_l) / len(X), len(X_r) / len(X)
                    e = p_l * calc_entropy(y_l) + p_r * calc_entropy(y_r)
                    if e < best_entropy:
                        best_entropy, best_d, best_v = e, d, v
        return best_d, best_v


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)),
    )
    X_new = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=1)

    y_predict = model.predict(X_new)
    z = y_predict.reshape(x0.shape)

    custom_cmap = colors.ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, z, cmap=custom_cmap)


if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)
    plot_decision_boundary(tree, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.show()