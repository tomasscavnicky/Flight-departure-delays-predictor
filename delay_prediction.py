#!/usr/bin/python2

from __future__ import print_function

import csv
import numpy as np

from pprint import pprint
from datetime import datetime
from collections import defaultdict


apts_fname = 'airports.csv'
in_fname = 'delays_dataset_clean.csv'
datefmt = '%Y-%m-%d %H:%M:%S'


linecount = 0
'''with open(in_fname) as data_file:
    for linecount,line in enumerate(data_file):
        pass
'''
linecount = 10000
# read first 100 lines, compute avg line len;
# linecount = int( 1.02 * (filesize / avg line len) )


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
            for linecount,_ in enumerate(self.f):
                pass
            batchsize = linecount
        
        self.batch = np.ndarray((batchsize, 10), dtype=np.float32)


    def __iter__(self):
        self.csvit = csv.DictReader(self.f)
        self.carriers = Counter()
        return self


    def next(self):
        b = self.batch
        b[:] = 0

        for i in xrange(len(b)):
            try:
                row = next(self.csvit)
            except StopIteration:
                if i == 0:
                    raise
                break

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

        return b[:i+1]


if __name__=='__main__':

    it = CsvReader(in_fname, 100000)
    for b in it:
        print(b)
        raw_input()



