import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def main():
    data = pd.read_csv('../data/creditcard.csv')

    scaler = StandardScaler()
    data['amount_z'] = scaler.fit_transform(data['Amount'].reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    print(data.head())

    print(data.describe())
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    # sampling


    # x_undersample, y_undersample  = generate_balanced_sample_manual(data)
    x_undersample, y_undersample = generate_balanced_sample_rus(X, y)


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
    main()
