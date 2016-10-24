import pandas as pd
import math
import time
import numpy as np

'''Find and replace NaN values'''
def est_nan(data, target_feature, reference_feature):

    plotting = False  # Show plots for data estimation where missing values were found

    # Max number of values to use for ratio
    tail_n = 100

    # make sure there are values for first and last rows
    if (pd.isnull(data[target_feature].iloc[-1])):
        print('NaN values at end of data with length: ' + str(len(data)))
        trim_at = data[target_feature].iloc[:(len(data) - 1)].last_valid_index()
        row_drop_num = len(data) - trim_at
        print('Dropping %d rows' % row_drop_num)
        data = data.drop(data.index[trim_at: -1])
        print('New length of dataset: ' + str(len(data)))

    if (pd.isnull(data[target_feature].iloc[0])):
        print('NaN values at beginning of data with length: ' + str(len(data)))
        trim_at = data[target_feature].iloc[0:].first_valid_index()
        row_drop_num = trim_at
        print('Dropping %d rows' % row_drop_num)
        data = data.drop(data.index[0: trim_at])
        print('New length of dataset: ' + str(len(data)))

    # find indexes of NaNs in A and B columns and create arrays
    nanindex = data.index[data[target_feature].apply(np.isnan)]
    valIndex = data.index[data[target_feature].apply(np.isfinite)]
    valAIndex = data.index[data[reference_feature].apply(np.isfinite)]
    dualIndex = data.index[data[target_feature].apply(np.isfinite) & data[reference_feature].apply(np.isfinite)]

    df_index = data.index.values.tolist()
    nindex = [df_index.index(i) for i in nanindex]
    # valArray = [df_index.index(i) for i in valIndex]

    # bcRatio set as 1, unless using Coindesk values to fill in NaNs
    try:
        # sum the last 100 values (~2 hours) of ticker data to get the conversion rate
        bcRatio = (
        sum(data[target_feature].ix[dualIndex].tail(tail_n)) / sum(data[reference_feature].ix[dualIndex].tail(tail_n)))
    except:
        bcRatio = 1

    # Find nearest value function
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        else:
            return array[idx]

    nanStart = 0
    nanEnd = 0
    prevNanIndex = -1
    for n in range(len(nindex)):

        # Indices of NaN array
        n_i_1t = (nindex[n] - 1)
        n_i_t = nindex[n]
        n_i_t1 = (nindex[n] + 1)

        # Values of NaN Array
        n_v_1t = data.ix[n_i_1t][reference_feature]

        # If the last value in the data array is NaN
        # and the next value is not NaN
        if (prevNanIndex == n_i_1t) & (n_i_t1 not in nindex):

            # The NaN Series ends with the next non NaN index
            nanEnd = n_i_t1
            placeholder = float(data.loc[nanStart, target_feature])

            # The number of NaN values in the series
            nanDiff = nanEnd - (nanStart + 1)

            # The averaged difference in values between start of NaN series and end of NaN Series
            diff = (data.ix[nanEnd][target_feature] - data.ix[nanStart][target_feature]) / (nanDiff + 1)

            # For each NaN in series, replace with scaled value
            for i in range(nanDiff):

                # Local index of NaN series
                r = i + 1
                # Global index of the dataframe
                row_index = nanStart + r

                # Find the nearest value to serve as reference
                nearestA = find_nearest(valAIndex, (row_index))
                nearestB = find_nearest(valIndex, (row_index))
                nnA = abs(nearestA - row_index)
                nnB = abs(nearestB - row_index)

                if (nnB <= nnA):

                    # Increment by the averaged difference
                    increment = r * diff
                    estimated = (placeholder + increment)
                    data.loc[row_index, target_feature] = estimated

                else:
                    # If A is closer use the conversion rate to port over values
                    placeholderA = data.loc[nearestA, reference_feature]
                    estimated = placeholderA * float(bcRatio)
                    data.loc[row_index, target_feature] = estimated

            # Reset Series Variables
            nanStart = 0
            nanEnd = 0
            prevNanIndex = -1

        # If the last value was NaN and so is the next
        elif (prevNanIndex == n_i_1t) & (n_i_t1 in nindex):
            pass

        # If the last value is not NaN, but the next is, mark the start index
        elif (n_i_1t not in nindex) & (n_i_t1 in nindex):
            nanStart = n_i_1t

        # If only one NaN is found isolated, use the preceding and folling values to fill it in
        elif (n_i_t1 not in nindex) & (n_i_t1 not in nindex):
            nanDiff = n_i_t1 - (n_i_1t + 1)
            placeholder = float(data.loc[n_i_1t, target_feature])
            diff = (data.ix[n_i_t1][target_feature] - data.ix[n_i_1t][target_feature]) / float(nanDiff + 1)
            row_index = n_i_t
            estimated = (data.ix[n_i_1t][target_feature] + diff) * bcRatio
            data.loc[row_index, target_feature] = estimated

            # Reset Series Variables
            nanStart = 0
            nanEnd = 0
            prevNanIndex = -1

        else:
            print("Error matching NaN series")
            nanStart = n_i_1t

        # Set the index of the last NaN to the current index
        prevNanIndex = nindex[n]

    if plotting == True:
        # print(data)
        plot_results(data.index, data[target_feature], data[reference_feature])

    return data


def replace_nans_noise(data, feature_columns):
    for col in range(len(feature_columns)):
        standard_deviation = data[feature_columns[col]].std(axis=0, skipna=True)
        mean_data = data[feature_columns[col]].mean(axis=0, skipna=True)
        data[feature_columns[col]] = [np.random.normal(mean_data, standard_deviation, 1)[0]
                                      if pd.isnull(data[feature_columns[col]].iloc[row])
                                      else data[feature_columns[col]].iloc[row]
                                      for row in range(len(data))]
    return data

# Plot results
def plot_results(X_plot, A_plot, B_plot):
    plt.plot(X_plot, A_plot, 'blue', alpha=0.5)
    plt.plot(X_plot, B_plot, 'red', alpha=0.5)

    plt.legend(loc='lower left')
    plt.show()