#!/usr/bin/env python

"""
This is the executable for the Ridge method, which is used in the Abed example.  

We run 10-fold cross validation on the training data and obtain predictions on 
the test folds. Next, we fit the model on the full training data and predict 
the data from the test dataset.

"""

import sys
import time

from six.moves import cPickle

from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge


def load_dataset(filename):
    """ Load the dataset from the filename. We assume it is a pickled Bunch """
    with open(filename, "rb") as fid:
        data = cPickle.load(fid)
    return data


def model_fit(X, y, alpha):
    """ Fit the model on the given X and y, while measuring fitting time. """
    clf = Ridge(alpha=alpha)
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


def run_cv(X, y, cv_seed, alpha):
    """ Run cross validation on the given training data.

    This function runs 10-fold CV on the given data, measuring total time 
    needed by the classifier, through the functions model_fit and 
    model_predict. Note that the predictions are stored in a simple list, which 
    can be compared to the true y (which is what Abed will do).
    """
    train_time = 0.0
    predictions = [0] * len(y)
    for trainidx, testidx in KFold(len(y), 10, random_state=cv_seed):
        X_train, X_test = X[trainidx], X[testidx]
        y_train = y[trainidx]

        model, fit_duration = model_fit(X_train, y_train, alpha)
        test_pred, pred_duration = model_predict(model, X_test)
        for idx, pred in zip(testidx, test_pred):
            predictions[idx] = pred
        train_time += fit_duration + pred_duration
    return predictions, train_time


def print_results(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    coef_true,
    coef_pred,
    total_time,
):
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


def main(train_filename, test_filename, cv_seed, alpha):
    """ Run cross validation fitting the model on the training data, fit the 
    model on the total training data, and use that to predict the test dataset.  
    Finally print the results to stdout.
    """
    alpha = float(alpha)
    print("# ridge, cost = %g" % alpha)

    train = load_dataset(train_filename)
    test = load_dataset(test_filename)
    y_train_pred, cv_time = run_cv(train.X, train.y, cv_seed, alpha)

    model, train_time = model_fit(train.X, train.y, alpha)

    y_test_pred, pred_time = model_predict(model, test.X)

    total_time = cv_time + train_time + pred_time

    print_results(
        train.y,
        y_train_pred,
        test.y,
        y_test_pred,
        train.true_coef,
        model.coef_,
        total_time,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
