from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as skflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from utils.data_loader import read_data, loader_data
from utils.data_splitter import split_data_sets
from utils.cleaner import est_nan, replace_nans_noise

plotting = False
log_dir = 'tmp/timeseries/' # directory for TensorBoard logging

# Data Files
file_01 = './datasets/file_01.csv'
file_02 = './datasets/file_02.csv'

# Hyperparameters
prediction = 12 # How many steps into the future to predict (12 = 5 min here)
steps_forward = prediction
steps_backward = 0  # value <= 0
inputs_default = 0
hidden = 128
batch_size = 256
n_steps = seq_len = 1 # number of elements in sequence to classify
epochs = 2000
test_sets = 6
test_size = 0.2

dnn_hidden = [12, 24, 48, 24, 12] # Hidden Layers in fully connected layer (if enabled)

# Test and validation set size if using validation set
test_size_val = 0.2
validation_size = 0.2

# LSTM Hyperparameters
learning_rate = 0.05
forget_bias = 0.8
keep_prob = 0.8 #0.9998

# Inputs and target value
X_columns = ['btce-buy','btce-volume','btce-low','btce-sell','btce-high',
            'threshold_setting','prev_tx_count_in', 'prev_mass_out',
            'prev_value_in','prev_mass_in', 'prev_value_out', 'prev_tx_count_out']
y_column = 'btce-price'
y_column_ref = 'cd-price' # This fills in any missing target data with data from a redundant data set

# Optional feature shifting
input_range = {}


t0 = time.time()

def lstm_model(X, y):
    
    X = tf.reshape(X, [-1, n_steps, n_input])  # shape: [batch_size, n_steps, n_input]
    X = tf.transpose(X, [1, 0, 2]) # shape: [n_steps, batch_size, n_input]
    X = tf.reshape(X, [-1, n_input])  # shape: [n_steps*batch_size, n_input]

    # Split data for sequences
    X = tf.split(0, n_steps, X)  # n_steps * (batch_size, n_input)

    init = tf.random_normal_initializer(batch_size, stddev=0.05)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden, forget_bias=forget_bias)
    
    # Dropout
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    output, _ = tf.nn.rnn(lstm_cell, X, dtype=tf.float32)

    # Fully connected layer with dropout
    output = tf.nn.dropout(
        layers.stack(output[0],
        layers.fully_connected,
        dnn_hidden),
        keep_prob
    )

    regression = skflow.models.linear_regression(output, y) # Use output[0] if omitting fully connected layer

    return regression

def data_from_csv(file_01, file_02):
    data = read_data(file_01, file_02)

    print('Cleaning NaN values from price graph...')
    data = est_nan(data, y_column, y_column_ref)
    data = replace_nans_noise(data, X_columns)
    print("done in %0.3fs." % (time.time() - t0))
    return data

t1 = time.time()
data = data_from_csv(file_01, file_02)

print('Generating training, test and validation sets...')
X, y = loader_data(source=data, y_column=y_column, X_columns=X_columns,
                           inputs_per_column=input_range, inputs_default=inputs_default,
                           steps_forward=prediction)

X_train, y_train, X_test, y_test = split_data_sets(X, y, test_size, test_sets)

print("done in %0.3fs." % (time.time() - t1))


#______________________________________________________________________#
#                              MODELS                                  #
#______________________________________________________________________#


t1 = time.time()
print('Building model..')

# number of steps
steps = (X_train.shape[0] / batch_size) * epochs
n_input = X_train.shape[1]
print('Number of features: {0}'.format(n_input))
      
X_train, y_train = X_train.astype(np.float32).copy(), y_train.astype(np.float32).copy()
X_test, y_test = X_test.astype(np.float32).copy(), y_test.astype(np.float32).copy()
X, y = X.astype(np.float32).copy(), y.astype(np.float32).copy()

model = skflow.TensorFlowEstimator(
    model_fn=lstm_model, 
    n_classes=0, # n_classes = 0 for regression 
    verbose=0,
    batch_size=batch_size, 
    steps=steps, 
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=learning_rate,
        l1_regularization_strength=0.001),
    config=skflow.RunConfig(save_checkpoints_secs=1)
    )

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:

    # TensorBoard Summary Writer
    summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

    model.fit(X_train, y_train, logdir=log_dir)

print("done in %0.3fs." % (time.time() - t1))

#______________________________________________________________________#
#                             RESULTS                                  #
#______________________________________________________________________#

t1 = time.time()
print('Predicting outputs...')

y_hat = model.predict(X)
y_train_predicted = model.predict(X_train)
train_rmse = sqrt(mean_squared_error(y_train, y_train_predicted))
y_test_predicted = model.predict(X_test)
test_rmse = sqrt(mean_squared_error(y_test, y_test_predicted))

train_err_abs = mean_absolute_error(y_train, y_train_predicted)
test_err_abs = mean_absolute_error(y_test, y_test_predicted)

print('Training Error: %.2f RMSE' % (train_rmse))
print('Test Error: %.2f RMSE' % (test_rmse))

print('Train Error Abs: %.2f Absolute Error' % (train_err_abs))
print('Test Error Abs: %.2f Absolute Error' % (test_err_abs))

loss_ = model.evaluate(X_test, y_test)
print('Loss: {0:f}'.format(loss_[0]['loss']))

print("done in %0.3fs." % (time.time() - t1))
print("Overall time: %0.3fs" % (time.time() - t0))

#______________________________________________________________________#
#                             PLOTTING                                 #
#______________________________________________________________________#

if plotting == True:
    plt.clf()
    X_plot = np.ravel(data['epoch'])[:-prediction]
    A_plot = np.ravel(y)
    B_plot = np.ravel(y_hat)

    plot_results(X_plot, A_plot, B_plot)

# Plot results
def plot_results(X_plot, A_plot, B_plot):
    plt.plot(X_plot, A_plot, 'blue', alpha=0.5)
    plt.plot(X_plot, B_plot, 'red', alpha=0.5)

    plt.legend(loc='lower left')
    plt.show()



