# -*- coding: UTF-8 -*-

# 引入numpy庫以進行數據處理
import numpy as np
# 引入operator庫以使用運算符
import operator
# 引入collections庫以使用計數器
import collections

"""
函數說明: 創建數據集

參數:
    無
返回:
    group - 數據集
    labels - 分類標籤
"""
def createDataSet():
    # 四組二維特徵
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四組特徵的標籤
    labels = ['愛情片', '愛情片', '動作片', '動作片']
    # 返回數據集和標籤
    return group, labels

"""
函數說明: kNN算法，分類器

參數:
    inX - 用於分類的數據(測試集)
    dataSet - 用於訓練的數據(訓練集)
    labels - 分類標籤
    k - kNN算法參數，選擇距離最小的k個點
返回:
    sortedClassCount[0][0] - 分類結果
"""
def classify0(inx, dataset, labels, k):
    # 計算距離
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5
    # k個最近的標籤
    k_labels = [labels[index] for index in dist.argsort()[0:k]]
    # 出現次數最多的標籤即為最終類別
    label = collections.Counter(k_labels).most_common(1)[0][0]
    # 返回分類結果
    return label

if __name__ == '__main__':
    # 創建數據集
    group, labels = createDataSet()
    # 測試集
    test = [101, 20]
    # kNN分類
    test_class = classify0(test, group, labels, 3)
    # 打印分類結果
    print(test_class)
