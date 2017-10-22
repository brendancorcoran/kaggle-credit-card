import itertools
from collections import Counter
from itertools import cycle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        1  # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.  #
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_precision_recall(recall, precision, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    # plt.fill_between(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
        average_precision))


def plot_precision_recall_thresholds(y, y_pred, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    colors = cycle(
        ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue', 'black'])

    plt.figure(figsize=(5, 5))

    j = 1
    for i, color in zip(thresholds, colors):
        y_test_predictions_prob = y_pred[:, 1] > i

        precision, recall, thresholds = precision_recall_curve(y, y_test_predictions_prob)

        # Plot Precision-Recall curve
        plt.plot(recall, precision, color=color,
                 label='Threshold: %s' % i)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example')
        plt.legend(loc="lower left")


def generate_undersample_km_rus(X: pd.DataFrame, y: pd.Series):
    rus = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def generate_undersample_rus(X: pd.DataFrame, y: pd.Series):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def generate_oversample_rus(X: pd.DataFrame, y: pd.Series):
    rus = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def generate_oversample_smote(X: pd.DataFrame, y: pd.Series):
    X_resampled, y_resampled = SMOTE().fit_sample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def generate_balanced_sample_manual(data: pd.DataFrame):
    # Number of data points in the minority class
    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)
    # Picking the indices of the normal classes
    normal_indices = data[data.Class == 0].index
    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    # Under sample dataset
    under_sample_data = data.iloc[under_sample_indices, :]
    x_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
    # Showing ratio
    print("Percentage of normal transactions: ",
          len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
    print("Percentage of fraud transactions: ",
          len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
    print("Total number of transactions in resampled data: ", len(under_sample_data))
    return x_undersample, y_undersample


def generate_train_test_split(X, y, test_size=0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    print("Number transactions train dataset: ", len(X_train))
    print("Number transactions test dataset: ", len(X_test))
    print("Total number of transactions: ", len(X_train) + len(X_test))
    return X_train, X_test, y_train, y_test