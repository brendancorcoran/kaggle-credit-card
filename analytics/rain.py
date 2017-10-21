import pprint
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_confusion_matrix, plot_precision_recall_thresholds


def main():
    data = load_data()

    scaler = StandardScaler()
    data['amount_z'] = scaler.fit_transform(data['Amount'].reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    print(data.head())

    print(data.describe())
    X, y = get_X_y(data)

    # sampling


    # x_undersample, y_undersample  = generate_balanced_sample_manual(data)
    x_undersample, y_undersample = generate_balanced_sample_rus(X, y)


def get_X_y(data):
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    return X, y


def load_data():
    data = pd.read_csv('../data/creditcard.csv')
    return data


def main_pipeline_proba():
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('classify', LogisticRegression(C=0.01, penalty='l1'))
    ])

    data = load_data()
    X, y = get_X_y(data)

    # TODO: move this into the pipeline
    X_undersample, y_undersample = generate_balanced_sample_rus(X, y)

    pipe.fit(X_undersample, y_undersample)

    # predict on source data
    y_pred_proba = pipe.predict_proba(X)

    y_pred_proba_1s = [x[1] for x in y_pred_proba]

    average_precision = average_precision_score(y, y_pred_proba_1s)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    plot_precision_recall_thresholds(y, y_pred_proba)

    # precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_1s)
    # plot_precision_recall(precision, recall, average_precision)

    plt.show()


def main_pipeline():
    pp = pprint.PrettyPrinter(indent=4)

    pipe = Pipeline([
        ('scale', StandardScaler()),
        # ('classify', LinearSVC())
        ('classify', LogisticRegression())
    ])

    data = load_data()
    X, y = get_X_y(data)

    # TODO: move this into the pipeline
    X_undersample, y_undersample = generate_balanced_sample_rus(X, y)

    pipe.fit(X_undersample, y_undersample)

    # predict on source data
    y_pred = pipe.predict(X)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()
    print('plt.show')

    # C_OPTIONS = [1, 10, 100, 1000]
    #
    # param_grid = [{
    #     'classify__C': C_OPTIONS
    # }]
    grid = GridSearchCV(pipe, cv=4, n_jobs=1, param_grid=param_grid)
    # grid.fit(X_undersample, y_undersample)
    #
    # pp.pprint(grid.cv_results_)


def generate_balanced_sample_rus(X: pd.DataFrame, y: pd.Series):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)
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


if __name__ == '__main__':
    # main_pipeline()
    main_pipeline_proba()
