import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def read_data(file_01, file_02):
    data_01= pd.read_csv(
        file_01,
        parse_dates={'timeline': ['btce-time_stamp']},
        infer_datetime_format=True)
    data_02 = pd.read_csv(
        file_02,
        parse_dates={'timeline': ['epoch_time_stamp']},
        infer_datetime_format=True)

    data_02 = data_02.drop_duplicates('epoch')
    data_01['timeline'] = data_01['timeline'].astype(float)
    data_02['timeline'] = data_02['timeline'].astype(float)

    data_ = data_02.set_index('timeline').reindex(data_01.set_index('timeline').index, method='nearest').reset_index()
    data = pd.merge(data_01, data_, on='timeline', suffixes=('_', ''))
    return data

def loader_data(source, y_column, X_columns, inputs_per_column=None, inputs_default=3, steps_forward=1):
    # Shift the target by the number of steps specified in prediction variable
    y = source[y_column].shift(-steps_forward)

    # Normalize data to mean and unit variance
    scaler = StandardScaler()
    new_X = pd.DataFrame(scaler.fit_transform(source[X_columns]), columns=X_columns)

    X = pd.DataFrame()

    for column in X_columns:
        inputs = inputs_per_column.get(column, None)
        if inputs:
            inputs_list = range(inputs[0], inputs[1]+1)
        else:
            inputs_list = range(-inputs_default, 1)

        for i in inputs_list:
            col_name = "%s_%s" % (column, i)
            X[col_name] = new_X[column].shift(-i)  # Note: shift direction is inverted

    X = pd.concat([X, y], axis=1)
    X.dropna(inplace=True, axis=0)
    y = X[y_column].reshape(X.shape[0], 1)
    X.drop([y_column], axis=1, inplace=True)

    return X.values, y
