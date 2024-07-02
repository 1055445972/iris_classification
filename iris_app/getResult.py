import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
# 定义最小距离分类器类
class MinimumDistanceClassifier:
    def __init__(self):
        self.centroids = None
    
    def fit(self, X, y):
        unique_classes = np.unique(y)
        self.centroids = {}
        
        for cls in unique_classes:
            # 计算每个类别的中心点
            class_samples = X[y == cls]
            centroid = np.mean(class_samples, axis=0)
            self.centroids[cls] = centroid
    
    def predict(self, X):
        y_pred = []
        
        for sample in X:
            distances = []
            
            # 计算样本点与各类别中心点的距离
            for centroid in self.centroids.values():
                dist = np.linalg.norm(sample - centroid)
                distances.append(dist)
            
            # 找到距离最小的类别作为预测结果
            min_dist_index = np.argmin(distances)
            predicted_class = list(self.centroids.keys())[min_dist_index]
            y_pred.append(predicted_class)
        
        return y_pred

from collections import Counter

# 定义k近邻分类器类
class KNearestNeighborsClassifier:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y_pred = []
        
        for sample in X:
            distances = []
            
            # 计算样本点与训练集中所有点的距离
            for train_sample, train_label in zip(self.X, self.y):
                dist = np.linalg.norm(sample - train_sample)
                distances.append((dist, train_label))
            
            # 根据距离排序，并选取前k个最近邻居
            k_nearest = sorted(distances)[:self.k]
            
            # 统计最近邻居的类别
            labels = [label for _, label in k_nearest]
            majority_vote = Counter(labels).most_common(1)[0][0]
            y_pred.append(majority_vote)
        
        return y_pred


class Node:
    def __init__(self, feature=None, threshold=None, label=None):
        self.feature = feature  # 用于划分的特征索引
        self.threshold = threshold  # 划分特征的阈值
        self.label = label  # 叶子节点的类别标签
        self.children = {}  # 子节点的字典，键为特征值，值为对应的子节点

class ID3Classifier:
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        y_pred = []
        for sample in X:
            label = self._traverse_tree(sample, self.root)
            y_pred.append(label)
        return y_pred

    def _build_tree(self, X, y):
        # 创建节点
        node = Node()

        # 如果所有样本都属于同一类别，则将节点标记为叶子节点
        if len(np.unique(y)) == 1:
            node.label = y[0]
            return node

        # 如果没有特征可用，则将节点标记为叶子节点，并选择出现次数最多的类别作为标签
        if X.shape[1] == 0:
            node.label = np.argmax(np.bincount(y))
            return node

        # 选择最佳特征和阈值进行划分
        best_feature, best_threshold = self._select_best_split(X, y)
        node.feature = best_feature
        node.threshold = best_threshold

        # 根据最佳划分进行子节点的递归构建
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
        node.children[0] = self._build_tree(X_left, y_left)
        node.children[1] = self._build_tree(X_right, y_right)

        return node

    def _select_best_split(self, X, y):
        best_info_gain = -1
        best_feature = None
        best_threshold = None

        num_features = X.shape[1]
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                info_gain = self._information_gain(X, y, feature, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)

        left_entropy = self._entropy(y_left)
        right_entropy = self._entropy(y_right)

        num_left = len(y_left)
        num_right = len(y_right)
        num_samples = num_left + num_right

        child_entropy = (num_left / num_samples) * left_entropy + (num_right / num_samples) * right_entropy
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-7))
        return entropy

    def _split_data(self, X, y, feature, threshold):
        mask = X[:, feature] <= threshold
        X_left = X[mask]
        y_left = y[mask]
        X_right = X[~mask]
        y_right = y[~mask]
        return X_left, y_left, X_right, y_right

    def _traverse_tree(self, sample, node):
        if node.label is not None:
            return node.label
        else:
            if sample[node.feature] <= node.threshold:
                return self._traverse_tree(sample, node.children[0])
            else:
                return self._traverse_tree(sample, node.children[1])



def get_confusion_matrix():
    # 加载鸢尾花数据集
    data = load_iris()
    X, y = data.data, data.target
    X = MinMaxScaler().fit_transform(X)
    
    # 创建并训练最小距离分类器
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,stratify=y)
    min_dist_classifier = MinimumDistanceClassifier()
    min_dist_classifier.fit(X_train, y_train)
    min_dist_pred = min_dist_classifier.predict(X_test)
    min_dist_accuracy = round(accuracy_score(y_test, min_dist_pred),2)
    min_dist_cm = confusion_matrix(y_test, min_dist_pred)

    # 创建并训练k近邻分类器
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,stratify=y)
    knn_classifier = KNearestNeighborsClassifier(k=3)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    knn_accuracy = round(accuracy_score(y_test, knn_pred),2)
    knn_cm = confusion_matrix(y_test, knn_pred)

    # 创建并训练ID3分类器
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,stratify=y)
    id3_classifier = ID3Classifier()
    id3_classifier.fit(X_train, y_train)
    id3_pred = id3_classifier.predict(X_test)
    id3_accuracy = round(accuracy_score(y_test, id3_pred),2)
    id3_cm = confusion_matrix(y_test, id3_pred)

    print("最小距离分类器的准确率:", min_dist_accuracy)
    print("k近邻分类器的准确率:", knn_accuracy)
    print("ID3分类器的准确率:", id3_accuracy)

    # 计算绘图数据
    value1 = [[j,2-i, int(min_dist_cm[i,j])] for i in range(3) for j in range(3)]
    value2 = [[j,2-i, int(knn_cm[i,j])] for i in range(3) for j in range(3)]
    value3 = [[j,2-i, int(id3_cm[i,j])] for i in range(3) for j in range(3)]

    labels = data.target_names
    return value1,value2,value3,min_dist_accuracy,knn_accuracy,id3_accuracy
