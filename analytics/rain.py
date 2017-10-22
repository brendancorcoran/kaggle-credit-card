import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, average_precision_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_confusion_matrix, plot_precision_recall_thresholds, generate_undersample_rus, \
    generate_oversample_smote, generate_train_test_split


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
    x_undersample, y_undersample = generate_undersample_rus(X, y)


def get_X_y(data):
    X = data.drop(['Time', 'Class'], axis=1)
    y = data['Class']
    return X, y


def load_data():
    data = pd.read_csv('../data/creditcard.csv')
    return data


def main_pipeline_proba_grid_rf():
    pipe = Pipeline([
        ('scale', StandardScaler()),
        # ('rf', RandomForestClassifier(random_state=0))
        ('rf', RandomForestClassifier(criterion='gini', bootstrap=False, random_state=0))
        # optimised from previous runs
    ])

    data = load_data()
    X, y = get_X_y(data)

    # X_sample, y_sample = generate_undersample_rus(X, y)
    # X_sample, y_sample = generate_undersample_km_rus(X, y)
    X_sample, y_sample = generate_oversample_smote(X, y)

    # generate test/train for the sample equalized
    X_train, X_test, y_train, y_test = generate_train_test_split(X_sample, y_sample, test_size=0.8)

    # gridsearch ov c param space
    parameters = {'rf__max_depth': list(range(1, 11, 2))
                  # 'rf__bootstrap': [True, False],
                  # 'rf__criterion': ['gini', 'entropy']
                  }
    # clf = GridSearchCV(pipe, parameters, scoring='average_precision')
    clf = GridSearchCV(pipe, parameters, scoring='f1')
    clf.fit(X_train, y_train)

    print('Best params found on training set:')
    print(clf.best_params_)
    print('...')
    # pp.pprint(clf.cv_results_)
    print('...')

    # predict on the original dataset (i.e. unbalanced)
    # TODO: apply this to the train_test_split wrapping
    y_pred_proba = clf.predict_proba(X)

    y_pred_proba_1s = [x[1] for x in y_pred_proba]

    average_precision = average_precision_score(y, y_pred_proba_1s)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    print('Classification report for each class:')
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))

    # plot...
    plot_precision_recall_thresholds(y, y_pred_proba)
    plt.show()


def main_pipeline_proba_grid():
    pp = pprint.PrettyPrinter(indent=4)

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('classify', LogisticRegression(penalty='l1'))
    ])

    data = load_data()
    X, y = get_X_y(data)

    X_undersample, y_undersample = generate_undersample_rus(X, y)

    # gridsearch ov c param space
    parameters = {'classify__C': [0.01, 0.1, 1]}
    clf = GridSearchCV(pipe, parameters, scoring='average_precision')
    # clf.fit(X_undersample, y_undersample)
    clf.fit(X, y)

    print('Best params found on training set:')
    print(clf.best_params_)
    print('...')
    # pp.pprint(clf.cv_results_)
    print('...')

    # predict on the original dataset (i.e. unbalanced)
    y_pred_proba = clf.predict_proba(X)

    y_pred_proba_1s = [x[1] for x in y_pred_proba]

    average_precision = average_precision_score(y, y_pred_proba_1s)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    print('Classification report for each class:')
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))

    # plot...
    plot_precision_recall_thresholds(y, y_pred_proba)
    plt.show()


def main_pipeline_proba():
    pipe = Pipeline([
        ('scale', StandardScaler()),
        # ('classify', LogisticRegression(C=0.01, penalty='l1'))
        ('rf', RandomForestClassifier(max_depth=5, random_state=0))
    ])

    data = load_data()
    X, y = get_X_y(data)

    X_undersample, y_undersample = generate_undersample_rus(X, y)
    # X_undersample, y_undersample = generate_oversample_rus(X, y)

    # generate test/train for the sample equalized
    X_train, X_test, y_train, y_test = generate_train_test_split(X_undersample, y_undersample)

    pipe.fit(X_train, y_train)

    # predict on the original dataset (unbalanced)
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
        ('classify', LogisticRegression(C=0.01, penalty='l1'))
    ])

    data = load_data()
    X, y = get_X_y(data)

    X_undersample, y_undersample = generate_undersample_rus(X, y)

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
    # grid = GridSearchCV(pipe, cv=4, n_jobs=1, param_grid=param_grid)
    # grid.fit(X_undersample, y_undersample)
    #
    # pp.pprint(grid.cv_results_)





if __name__ == '__main__':
    # main_pipeline()
    # main_pipeline_proba()
    # main_pipeline_proba_grid()
    main_pipeline_proba_grid_rf()
