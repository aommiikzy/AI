"""A module to for ITCS451 assignment 5."""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def load_data():
    """
    Return dataset for classification in a form of numpy array.

    The dataset should be in a 2D array where each column is a feature
    and each row is a data point. The last column should be the label.

    Your data should contain only number.
    If you have a categorical feature,  you can replace your feature with
    "one-hot encoding". For example, a feature 'grade' of values:
    'A', 'B', 'C', 'D', and 'F' can be replaced with 5 features:
    'is_grade_A', 'is_grade_B', ..., and 'is_grade_F'.

    For the label, you can replace them with 0, 1, 2, 3, ...

    Returns
    -----------
    dataset : np.array
        2D array containing features and label.

    """
    # TODO: 1
    # raise NotIxmplementedError
    # The following example is a dataset about
    # quiz, hw, reading hours and grade.
    # fullscore 5,15,10
    return np.array([
        [5, 15, 10, 4],  # 4 is A
        [5, 13, 7, 3],  # 3 is B
        [4, 10, 3, 2],  # 2 is C
        [3, 0, 4, 1],  # 1 is D
        [2, 1, 1, 0],  # 0 is F
        [5, 7, 8, 3],
        [5, 14, 10, 4],
        [1, 1, 1, 0],
        [1, 2, 2, 0],
        [0, 2, 2, 0],
        [2, 2, 1, 0],
        [1, 2, 1, 0],
        [4, 6, 4, 3],
        [5, 6, 3, 2],
        [4, 10, 9, 4],
        [4, 15, 9, 4],
        [3, 15, 9, 4],
        [4, 10, 8, 4],
        [5, 12, 7, 3],
        [5, 12, 9, 3],
        [2, 8, 8, 3],
        [3, 11, 7, 3],
        [3, 11, 10, 4],
        [3, 13, 5, 4],
        [5, 2, 3, 3],
        [4, 4, 5, 2],
        [5, 13, 6, 3],
        [5, 14, 8, 3],
        [5, 12, 8, 3],
        [4, 12, 2, 4],
        [3, 12, 4, 4],
        [4, 3, 5, 1],
        [3, 5, 6, 1]
    ])


def train(features, labels):
    """
    Return a decision tree model after "training".

    For more information on how to use Decision Tree, please visit
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Returns
    ---------
    tree : sklearn.tree.DecisionTreeClassifier

    """
    # TODO 2:
    # TempFeatures = features
    # TempLabels = labels
    # AfterTraining = DecisionTreeClassifier().fit(TempFeatures, TempLabels)

    return DecisionTreeClassifier().fit(features, labels)


def predict(model, features):
    """
    Return the prediction result.

    Parameters
    --------
    model : sklearn.tree.DecisionTreeClassifier
    features : np.array

    Returns
    -------
    predictions : np.array

    """
    # TODO 3:
    # raise NotImplementedError
    return model.predict(features)


def evaluate(labels, predictions, label_types):
    """
    Return the confusion matrix.

    Parameters
    ---------
    labels : np.array
    predictions : np.array
    label_types : np.array
        This is a list that keeps unique values of the labels, so that
        the confusion matrix has the same order.

    Returns
    ---------
    confusion_matrix : np.array
        The array should have the shape of [# of classes, # of classes].
        Number of class is the length of `label_types`.

    """
    # TODO 4:
    # You can use a library if you can find it.
    # raise NotImplementedError
    return confusion_matrix(labels, predictions, label_types)


if __name__ == "__main__":
    data = load_data()
    all_labels = data[:, -1]
    all_labels = set(all_labels)
    all_labels = np.array([v for v in all_labels])
    train_data, test_data = train_test_split(data,test_size=0.3)
    features = train_data[:, :-1]
    labels = train_data[:, -1]
    tree = train(features, labels)
    predictions = predict(tree, features)
    print('Training Confusion Matrix:')
    print(evaluate(labels, predictions, all_labels))
    predictions = predict(tree, test_data[:, :-1])
    print('Testing Confusion Matrix:')
    print(evaluate(test_data[:, -1], predictions, all_labels))
