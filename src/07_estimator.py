#!/usr/bin/env python
'''
based on tutorials on https://www.tensorflow.org/get_started/estimator
'''

import os
from urllib.request import urlopen

import numpy as np
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris')

IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

MODEL_DIR = '/tmp/iris_model'

FEATURE_LENGTH = 4
CLASS_NUM = 3


def prepare_data():
    '''Download dataset if not available'''
    if not os.path.exists(DATA_DIR):
        print('mkdir {}'.format(DATA_DIR))
        os.makedirs(DATA_DIR)

    local_iris_training_file = os.path.join(DATA_DIR, IRIS_TRAINING)
    if not os.path.exists(local_iris_training_file):
        print('Downloading {}'.format(IRIS_TRAINING))
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(local_iris_training_file, 'wb') as f:
            f.write(raw)

    local_iris_test_file = os.path.join(DATA_DIR, IRIS_TEST)
    if not os.path.exists(local_iris_test_file):
        print('Downloading {}'.format(IRIS_TEST))
        raw = urlopen(IRIS_TEST_URL).read()
        with open(local_iris_test_file, 'wb') as f:
            f.write(raw)

    return (local_iris_training_file, local_iris_test_file)


def load_data(training_csv, test_csv):
    'Load dataset from csv file'
    # Each csv file looks like:
    #   30,4,setosa,versicolor,virginica
    #   5.9,3.0,4.2,1.5,1
    #   6.9,3.1,5.4,2.1,2
    #   ...
    # The first labels are:
    #   [The number of samples],[The number of features],[label1],[label2],[label3]
    # The content lines are:
    #   [feature0],[feature1],[feature2],...,[class]
    #
    # `load_csv_with_header` returns Dataset object.
    # `target` means the label data and features mean features.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=training_csv, target_dtype=np.int, features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=test_csv, target_dtype=np.int, features_dtype=np.float32)
    return (training_set, test_set)


def build_classifier():
    'build classifier'
    feature_columns = [
        tf.feature_column.numeric_column('x', shape=[FEATURE_LENGTH])
    ]
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=CLASS_NUM,
        model_dir=MODEL_DIR)


def build_input(training_set, test_set):
    'Build training input for estimator'
    training_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)
    return (training_input_fn, test_input_fn)


def train_classifier(classifier, input_fn):
    'Train classifier'
    print('Training data')
    classifier.train(input_fn=input_fn, steps=2000)
    # steps=2000 equals to call steps=1000 twice.
    # classifier.train(input_fn=input_fn, steps=2000)
    # = classifier.train(input_fn=input_fn, steps=1000)
    #   classifier.train(input_fn=input_fn, steps=1000)



def evaluate_classifier_by_test_input(classifier, test_input_fn):
    result = classifier.evaluate(input_fn=test_input_fn)
    print('Evaluated result')
    # result has 'loss' field and so on as well as 'accuracy' field.
    print(result)
    return result['accuracy']


def predict_data(classifier, new_samples):
    'Predict new sample'
    print('Predicting data')
    new_samples_np_array = np.array(new_samples, dtype=np.float32)
    # Do not specify y arguments in prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': new_samples_np_array}, num_epochs=1, shuffle=False)
    return list(classifier.predict(input_fn=predict_input_fn))


def main():
    'main entry point'
    (training_csv, test_csv) = prepare_data()
    (training_set, test_set) = load_data(training_csv, test_csv)
    classifier = build_classifier()
    (training_input_fn, test_input_fn) = build_input(training_set, test_set)
    train_classifier(classifier, training_input_fn)
    accuracy = evaluate_classifier_by_test_input(classifier, test_input_fn)
    print('Test Accuracy: {:f}'.format(accuracy))
    predictions = predict_data(classifier,
                               [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]])
    print(predictions)
    predicted_classes = [p['classes'] for p in predictions]
    print('New samples: Class Predictions: {}'.format(predicted_classes))


if __name__ == '__main__':
    main()
