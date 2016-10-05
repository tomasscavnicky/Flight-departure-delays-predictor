#!/usr/bin/python2

from __future__ import print_function
import sys
import numpy as np
from pprint import pprint
import csv
from datetime import datetime

apts_fname = 'airports.csv'
in_fname = 'delays_dataset.csv'
datefmt = '%Y-%m-%d %H:%M:%S'


airport2gps = {}
with open(apts_fname) as apts_csvfile:
    reader = csv.reader(apts_csvfile)
    for row in reader:
        a_code = row[4]
        if a_code:
            latitude, longitude = float(row[6]), float(row[7])
            airport2gps[a_code] = latitude, longitude



linecount = 0
'''with open(in_fname) as data_file:
    for linecount,line in enumerate(data_file):
        pass
'''
linecount = 10000
# read first 100 lines, compute avg line len;
# linecount = int( 1.02 * (filesize / avg line len) )


carriers_id = np.chararray(linecount, 3)
flight_numbers = np.zeros(linecount, dtype=np.uint16)
orig = np.ndarray((linecount, 2), dtype=np.float32)
dest = np.ndarray((linecount, 2), dtype=np.float32)
date = np.ndarray((linecount, 4), dtype=np.int16)
delays = np.zeros(linecount, dtype=np.int16)

carriers_set = set()
counter = 0

with open(in_fname) as data_file:
    reader = csv.DictReader(data_file)
    i = 0
    for row in reader:
        if i == linecount:
            break
        try:
            carrier = row['carrier']
            carriers_set.add(carrier)

            schdep = datetime.strptime(row['scheduled_departure'], datefmt)
            actdep = datetime.strptime(row['actual_departure'],    datefmt)

            delay = int( (actdep - schdep).total_seconds() / 60 )

            carriers_id[i] = carrier
            flight_numbers[i] = int(row['fltno'])
            orig[i] = airport2gps[row['dep_apt']]
            dest[i] = airport2gps[row['arr_apt']]
            date[i] = schdep.month, schdep.day, schdep.isoweekday(), schdep.hour * 60 + schdep.minute
            delays[i] = delay
            i += 1

        except Exception:
            pass

print('Error count:', linecount - i)


carrier_id_to_idx = {cid:idx for idx,cid in enumerate(carriers_set)}
n_carriers = len(carriers_set)
del carriers_set

def make_entry(i):
    carrier_idx = carrier_id_to_idx[carriers_id[i]]
    # one-hot carrier
    res = [0] * n_carriers
    res[carrier_idx] = 1
    # rest of data
    res.append(flight_numbers[i])
    res.extend(orig[i])
    res.extend(dest[i])
    res.extend(date[i])

    return np.array(res, dtype=np.float32), delays[i]  # float err !


for i in range(10):
    e,d = make_entry(i)
    print(i+1, e[n_carriers:])

