#!/usr/bin/python2

from __future__ import print_function

import sys
import csv
import numpy as np

from pprint import pprint
from datetime import datetime
from collections import defaultdict

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import keras.backend as k
from random import shuffle

in_fname = sys.argv[1] if len(sys.argv) > 1 else 'data/delays_dataset_clean.csv'
apts_fname = 'data/airports.csv'
datefmt = '%Y-%m-%d %H:%M:%S'


airport2gps = {}
with open(apts_fname) as apts_csvfile:
    reader = csv.reader(apts_csvfile)
    for row in reader:
        a_code = row[4]
        if a_code:
            latitude, longitude = float(row[6]), float(row[7])
            airport2gps[a_code] = latitude, longitude


class Counter(defaultdict):

    def __init__(self, *args, **kwargs):
        self.i = 0

    def __missing__(self, key):
        val = self[key] = self.i
        self.i += 1
        return val


class CsvReader:

    def __init__(self, csv_fname, batchsize=None):
        self.f = open(csv_fname)

        if batchsize is None:
            linecount = 0
            for _ in self.f:
                linecount += 1
            self.f.seek(0)
            
            # linecount = 500000
            batchsize = linecount
        
        self.batch = np.ndarray((batchsize, 10), dtype=np.float32)
        self.batch_targets = np.zeros(batchsize, dtype=np.int16)


    def __iter__(self):
        self.csvit = csv.DictReader(self.f)
        self.carriers = Counter()
        return self


    def next(self):
        b = self.batch
        t = self.batch_targets
        b[:] = 0
        t[:] = 0

        errcnt = 0

        i, end = 0, len(b)
        while i < end:

            if i % 100000 == 0:
                print(i)
            
            try:
                row = next(self.csvit)

                cidx  = self.carriers[row['carrier']]
                fltno = int(row['flight_number'])
                orig  = airport2gps[row['dep_apt']]
                dest  = airport2gps[row['arr_apt']]

                schdep = datetime.strptime(row['scheduled_departure'], datefmt)
                actdep = datetime.strptime(row['actual_departure'],    datefmt)
                date   = schdep.month, schdep.day, schdep.isoweekday(), schdep.hour * 60 + schdep.minute
                delay  = int( (actdep - schdep).total_seconds() / 60 )

                entry = [cidx, fltno]
                entry.extend(orig)
                entry.extend(dest)
                entry.extend(date)

                b[i] = entry
                t[i] = delay

                i += 1

            except StopIteration:
                if i == 0:
                    raise
                break

            except Exception:
                errcnt += 1


        return b[:i], t[:i]


class decay_lr(Callback):
    ''' 
        n_epoch = no. of epochs after decay should happen.
        decay = decay value
    '''  
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.get_value()
        if epoch > 1 and epoch%self.n_epoch == 0 :
            new_lr= self.decay*old_lr
            k.set_value(self.model.optimizer.lr, new_lr)
        else:
            k.set_value(self.model.optimizer.lr, old_lr)


if __name__=='__main__':
    num_epochs = 20
    batchsize = 64
    it = CsvReader(in_fname, batchsize=1000000)
    data = next(iter(it))
    X, y = data


    train_X = X[:0.9*len(X)]
    train_Y = y[:0.9*len(y)]
    test_X = X[0.9*len(X):]
    test_Y = y[0.9*len(y):]

    model = Sequential()
    model.add(Dense(input_dim=len(train_X[0]), output_dim=100))   # , W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(1))                                     # , W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    tb_callback = TensorBoard(log_dir='./logs/logs_1', histogram_freq=0, write_graph=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    model.compile(optimizer='adagrad', loss='mean_absolute_error') #mean_absolute_error
    print('\nmodel summary:')
    print(model.summary())

    model.fit(train_X, train_Y, batchsize, num_epochs, verbose=1, shuffle=True, callbacks=[tb_callback, early_stop])

    rand_indices = range(len(test_X))
    shuffle(rand_indices)
    rand_indices = rand_indices[:100]
    for i in rand_indices:
        print(test_X)
        result = model.predict([test_X[i]], verbose=1)
        print('expected:', test_Y[i])
        print('predicted:', result)


