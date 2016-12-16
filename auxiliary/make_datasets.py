#!/usr/bin/env python

"""
This is part of the Abed Example. See LINK HERE for more info.

This program generates example datasets for use in the example.

"""

import os
from random import random
from six.moves import cPickle

from sklearn.datasets import make_regression
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import train_test_split

N_DATASETS = 10
DATADIR = '../datasets/'

def main():
    for i in range(N_DATASETS):
        bias = 10.0 * random()
        X, y, coef = make_regression(n_samples=900, n_features=20, 
                n_informative=10, bias=bias, noise=2.0, coef=True, 
                random_state=round(random()*1e6))
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                test_size=1.0/3.0, random_state=42)

        train = Bunch(X=X_train, y=y_train, true_coef=coef)
        test = Bunch(X=X_test, y=y_test)

        train_filename = os.path.join(DATADIR, 'dataset_%i_train.txt' % (i+1))
        test_filename = os.path.join(DATADIR, 'dataset_%i_test.txt' % (i+1))
        with open(train_filename, 'wb') as fid:
            cPickle.dump(train, fid, 2)
        with open(test_filename, 'wb') as fid:
            cPickle.dump(test, fid, 2)


if __name__ == '__main__':
    main()
