# kNN-simple-classfication

This repository demonstrates a simple implementation of the k-Nearest Neighbors (kNN) algorithm in Python. The program is designed to classify a new data point based on its distance to existing labeled data points.

## Project Overview

This project provides a basic implementation of the kNN algorithm, a popular and intuitive machine learning method used for classification tasks. Given a set of labeled data points, the algorithm calculates the distances between the new data point and existing data points, selecting the k closest neighbors. The most common label among these neighbors is used to classify the new data point.

### Files

- `knn_example.py`: The main Python script implementing the kNN algorithm.

## Code Explanation

### `createDataSet()`

This function creates a simple dataset with 4 two-dimensional points and their associated labels. The labels in this case are either `'愛情片'` (romantic movie) or `'動作片'` (action movie).

### `classify0(inx, dataset, labels, k)`

This function implements the kNN algorithm:
- **inx**: The input data point (testing set).
- **dataset**: The training data set.
- **labels**: The classification labels of the training data.
- **k**: The number of nearest neighbors to consider for the classification.
The function returns the label that occurs most frequently among the k nearest neighbors.

### How It Works
1. The function computes the Euclidean distance between the test point and all points in the dataset.
2. The distances are sorted in ascending order, and the labels of the k nearest neighbors are selected.
3. The label that appears most frequently among the k neighbors is assigned as the classification of the test point.

### Example Usage

In the example provided, the dataset consists of 4 points with two labels: `'愛情片'` and `'動作片'`. A test data point `[101, 20]` is classified using the kNN algorithm with `k=3`.

```python
if __name__ == '__main__':
    # Create dataset
    group, labels = createDataSet()
    
    # Define test point
    test = [101, 20]
    
    # Perform kNN classification
    test_class = classify0(test, group, labels, 3)
    
    # Output classification result
    print(test_class)  # Output: 動作片
