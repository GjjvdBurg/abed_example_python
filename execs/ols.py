#!/usr/bin/env python

"""
This is the executable for OLS, which is used in the Abed example.

Since we're not optimizing a hyperparameter, we can simply fit the model once 
and predict the test data.

"""

import sys
import time

from six.moves import cPickle

from sklearn.linear_model import LinearRegression

def load_dataset(filename):
    """ Load the dataset from the filename. We assume it is a pickled Bunch """
    with open(filename, 'rb') as fid:
        data = cPickle.load(fid)
    return data

def model_fit(X, y):
    """ Fit the model on the given X and y, while measuring fitting time. """
    clf = LinearRegression()
    start_t = time.time()
    clf.fit(X, y)
    duration = time.time() - start_t
    return clf, duration


def model_predict(clf, X):
    """ Predict the outcome for data X given the data, while measuring time """
    start_t = time.time()
    y_pred = clf.predict(X)
    duration = time.time() - start_t
    return y_pred, duration


def print_results(y_train_true, y_train_pred, y_test_true, y_test_pred, 
        coef_true, coef_pred, total_time):
    """ Print the results to stdout for Abed to pick up. Vector results for 
    which a true and a predicted value exist are printed side by side. Scalar 
    values such as running time are simply printed on a line.
    """

    print("% y_train_true y_train_pred")
    for yt, yp in zip(y_train_true, y_train_pred):
        print("%.16f %.16f" % (yt, yp))

    print("% y_test_true y_test_pred")
    for yt, yp in zip(y_test_true, y_test_pred):
        print("%.16f %.16f" % (yt, yp))

    print("% coef_true coef_pred")
    for ct, cp in zip(coef_true, coef_pred):
        print("%.16f %.16f" % (ct, cp))

    print("% total_time")
    print("%.16f" % total_time)


def main(train_filename, test_filename):
    """ Fit the model on the total training data, and use that to predict the 
    test dataset. We also obtain predictions of the outcome of the training 
    dataset, for consistency of the output format between methods.

    """
    print("# ols")

    train = load_dataset(train_filename)
    test = load_dataset(test_filename)

    model, train_time = model_fit(train.X, train.y)
    y_train_pred, pred_time_train = model_predict(model, train.X)
    y_test_pred, pred_time_test = model_predict(model, test.X)

    total_time = train_time + pred_time_train + pred_time_test

    print_results(train.y, y_train_pred, test.y, y_test_pred, train.true_coef, 
            model.coef_, total_time)


if __name__ == '__main__':
    main(*sys.argv[1:])
