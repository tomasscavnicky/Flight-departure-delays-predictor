#!/usr/bin/python2

from __future__ import print_function
import numpy as np
from pprint import pprint
import csv
from datetime import datetime

airport2gps = {}
with open('airports.csv') as airport_csvfile:
    reader = csv.reader(airport_csvfile)
    for row in reader:
        a_code = row[4]
        if a_code:
            latitude, longitude = float(row[6]), float(row[7])
            airport2gps[a_code] = latitude, longitude



linecount = 0
with open('2008.csv') as data_file:
    for line in data_file:
        linecount += 1
linecount = 1000000


carriers_id = np.chararray(linecount, 3)
flight_numbers = np.zeros(linecount, dtype=np.uint16)
orig = np.ndarray((linecount, 2), dtype=np.float32)
dest = np.ndarray((linecount, 2), dtype=np.float32)
date = np.ndarray((linecount, 4), dtype=np.int16)
delays = np.zeros(linecount, dtype=np.int16)

carriers_set = set()
counter = 0

with open('test_flight_info.csv') as data_file:
    reader = csv.DictReader(data_file)
    for i, row in enumerate(reader):
        if i == linecount:
            break
        try:
            carrier = row['UniqueCarrier']
            carriers_set.add(carrier)

            day   = int(row['DayofMonth'])
            month = int(row['Month'])
            year  = int(row['Year'])

            sch_dep_time = row['CRSDepTime']
            hr, mn = int(sch_dep_time[:-2]), int(sch_dep_time[-2:])
            schdep = datetime(year, month, day, hr, mn)

            act_dep_time = row['DepTime']
            hr, mn = int(act_dep_time[:-2]), int(act_dep_time[-2:])
            actdep = datetime(year, month, day, hr, mn)

            delay = int( (actdep - schdep).total_seconds() / 60 )
            if delay < -12*60:
                delay += 24*60

            carriers_id[i] = carrier
            flight_numbers[i] = int(row['FlightNum'])
            orig[i] = airport2gps[row['Origin']]
            dest[i] = airport2gps[row['Dest']]
            date[i] = schdep.month, schdep.day, schdep.isoweekday(), schdep.hour * 60 + schdep.minute
            delays[i] = delay

        except:
            counter += 1

print('Error count:', counter)


carrier_id_to_idx = {cid:idx for idx,cid in enumerate(carriers_set)}
n_carriers = len(carriers_set)
del carriers_set


def make_entry(i):
    carrirer_idx = carrier_id_to_idx[carriers_id[i]]
    # one-hot carrier
    res = [0] * n_carriers
    res[carrier_idx] = 1
    # rest of data
    res.append(flight_numbers[i])
    res.extend(orig[i])
    res.extend(dest[i])
    res.extend(date[i])

    return np.array(res, dtype=np.float32), delays[i]


pprint(carriers)

