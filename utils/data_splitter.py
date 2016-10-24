import pandas as pd
import numpy as np

# Split the dataset into training and test sets
def split_data_sets(X, y, test_size, sets=1):
    set_length = X.shape[0]/sets
    offset = 0
    X_train_lst = []
    y_train_lst = []
    X_test_lst = []
    y_test_lst = []
    for i in range(sets+1):
        offset = i*set_length
        train_length = int(set_length*(1-test_size))
        train_start = offset
        train_end = offset + train_length
        test_start = train_end
        test_end = (i+1)*set_length
        X_train_lst.append(X[train_start:train_end])
        y_train_lst.append(y[train_start:train_end])
        X_test_lst.append(X[test_start:test_end])
        y_test_lst.append(y[test_start:test_end])
    X_train = np.concatenate(X_train_lst)
    y_train = np.concatenate(y_train_lst)
    X_test = np.concatenate(X_test_lst)
    y_test = np.concatenate(y_test_lst)
    return X_train, y_train, X_test, y_test

# Split into training, test and validation sets
# To do: combine with function above
def split_data_sets_with_validation(X, y, test_size, validation_size, sets=1):
    set_length = X.shape[0]/sets
    offset = 0
    X_train_lst = []
    y_train_lst = []
    X_test_lst = []
    y_test_lst = []
    X_validation_lst = []
    y_validation_lst = []
    for i in range(sets+1):
        offset = i*set_length
        train_length = int(set_length*(1-(test_size + validation_size)))
        train_start = offset
        train_end = offset + train_length
        test_length = int(set_length*test_size)
        test_start = train_end
        test_end = train_end + test_length
        validation_start = test_end
        validation_end = (i+1)*set_length
        X_train_lst.append(X[train_start:train_end])
        y_train_lst.append(y[train_start:train_end])
        X_test_lst.append(X[test_start:test_end])
        y_test_lst.append(y[test_start:test_end])
        X_validation_lst.append(X[validation_start:validation_end])
        y_validation_lst.append(y[validation_start:validation_end])
    X_train = np.concatenate(X_train_lst)
    y_train = np.concatenate(y_train_lst)
    X_test = np.concatenate(X_test_lst)
    y_test = np.concatenate(y_test_lst)
    X_validation = np.concatenate(X_validation_lst)
    y_validation = np.concatenate(y_validation_lst)
    return X_train, y_train, X_test, y_test, X_validation, y_validation
