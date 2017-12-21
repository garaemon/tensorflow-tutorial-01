#!/usr/bin/env python
'''
See https://www.tensorflow.org/get_started/input_fn

Define special input function for tf.estimator.
'''

import itertools
import os
from urllib.request import urlopen

import pandas as pd
import numpy as np
import tensorflow as tf

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'boston_housing')
BOSTON_TRAINING = 'boston_train.csv'
BOSTON_TRAINING_URL = 'http://download.tensorflow.org/data/boston_train.csv'
BOSTON_TEST = 'boston_test.csv'
BOSTON_TEST_URL = 'http://download.tensorflow.org/data/boston_test.csv'
BOSTON_PREDICT = 'boston_predict.csv'
BOSTON_PREDICT_URL = 'http://download.tensorflow.org/data/boston_predict.csv'
COLUMNS = [
    "crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"
]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def prepare_dataset():
    if not os.path.exists(DATA_DIR):
        print('making directory {}'.format(DATA_DIR))
        os.makedirs(DATA_DIR)
    local_training_file = os.path.join(DATA_DIR, BOSTON_TRAINING)
    if not os.path.exists(local_training_file):
        raw = urlopen(BOSTON_TRAINING_URL).read()
        with open(local_training_file, 'wb') as f:
            f.write(raw)
    local_test_file = os.path.join(DATA_DIR, BOSTON_TEST)
    if not os.path.exists(local_test_file):
        raw = urlopen(BOSTON_TEST_URL).read()
        with open(local_test_file, 'wb') as f:
            f.write(raw)
    local_predict_file = os.path.join(DATA_DIR, BOSTON_PREDICT)
    if not os.path.exists(local_predict_file):
        raw = urlopen(BOSTON_PREDICT_URL).read()
        with open(local_predict_file, 'wb') as f:
            f.write(raw)

    # read csv as DataFrame
    training_data = pd.read_csv(
        local_training_file, skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_data = pd.read_csv(
        local_test_file, skipinitialspace=True, skiprows=1, names=COLUMNS)
    predict_data = pd.read_csv(
        local_predict_file, skipinitialspace=True, skiprows=1, names=COLUMNS)

    return (training_data, test_data, predict_data)


def my_input_fn(data_set, num_epochs=None, shuffle=True):
    'Return pair of a mapping of {name: feature} and tensor and label tensor'
    # When using numpy array as input_fn, we can use `tf.estimator.inputs.numpy_input_fn`:
    # my_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": np.array(x_data)} y=np.array(y_data), ...)
    #
    # When using pandas dataframe as input_fn, we can use `tf.estimator.inputs.pandas_input_fn`:
    # my_input_fn = tf.estimator.inputs.pandas_input_fn(
    #    x=pd.DataFrame({"x": x_data}), y=pd.Series(y_data), ...)
    #
    # `tf.SparseTensor` is good for sparse dataset.
    # tf.SparseTensor(indices=[[0, 1], [2, 4]] # non-zero indices
    #                 values=[6, 0.5], # values of non-zero indices respectedly
    #                 dense_shape=[3, 5]) # shape of tensor
    # => [[0, 6, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0.5]]
    #
    # pass input_fn to estimator as function
    #   classifier.train(input_fn=my_input_fn, steps=2000)
    # Use functools.partial to pass parameter to input function
    #   classifier.train(input_fn=partial(my_input_fn, data_set=training_set), steps=2000)
    #    ==> my_input_fn(training_set) will be called
    # Using lambda is the third option
    #  classifier.train(input_fn=lambda: my_input(training_set), steps=2000)
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)


def main():
    (training_data, test_data, predict_data) = prepare_dataset()
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        model_dir='/tmp/boston_model')
    regressor.train(input_fn=my_input_fn(training_data), steps=5000)
    ev = regressor.evaluate(input_fn=my_input_fn(test_data, num_epochs=1, shuffle=False))
    loss_score = ev['loss']
    print('Loss: {0:f}'.format(loss_score))
    y = regressor.predict(
        input_fn=my_input_fn(predict_data, num_epochs=1, shuffle=False)
    )
    predictions = list(p['predictions'] for p in itertools.islice(y, 6))
    print('Predictions: {}'.format(str(predictions)))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()