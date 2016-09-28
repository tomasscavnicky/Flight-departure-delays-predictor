from __future__ import print_function
import numpy as np
from pprint import pprint
import csv
from datetime import datetime

airport2gps = {}

test_data = {}
carriers = {}

with open('airports.csv') as airport_csvfile:
    reader = csv.reader(airport_csvfile)
    for row in reader:
        a_code = row[4]
        if a_code:
            latitude = float(row[6])
            longitude = float(row[7])
            airport2gps[a_code] = latitude, longitude



linecount = 0
with open('2008.csv') as test_data:
    for line in test_data:
        linecount += 1

linecount = 1000000

flight_numbers = np.zeros(linecount, dtype=int)
orig = np.ndarray((linecount, 2), dtype=float)
dest = np.ndarray((linecount, 2), dtype=float)
sch_dept_date = np.ndarray((linecount, 4), int)

delays = np.zeros(linecount, dtype=int)

counter = 0

with open('test_flight_info.csv') as test_data:
    reader = csv.DictReader(test_data)
    for i, row in enumerate(reader):
        if i == linecount:
                break
        try:
            carriers[row['UniqueCarrier']] = True
            # print(row['UniqueCarrier'], row['FlightNum'], row['Origin'], row['Dest'], int(row['DepTime']), int(row['CRSDepTime']), int(row['Year']), int(row['Month']), int(row['DayofMonth']), int(row['DayOfWeek']))
            
            dep_time = row['CRSDepTime']
            hr, mn = int(dep_time[:-2]), int(dep_time[-2:])
            year = int(row['Year'])
            month = int(row['Month'])
            day = int(row['DayofMonth'])

            sch_departure = datetime(year, month, day, hr, mn, 0)

            act_dep_time = row['DepTime']
            hr, mn = int(act_dep_time[:-2]), int(act_dep_time[-2:])
            ac_departure = datetime(year, month, day, hr, mn, 0)

            delay = int((ac_departure - sch_departure).total_seconds())
            
            if delay < -12*60*60:
                delay += 24*60*60

            flight_numbers[i] = int(row['FlightNum'])
            orig[i] = np.array(list(airport2gps[row['Origin']]))
            dest[i] = np.array(list(airport2gps[row['Dest']]))
            sch_dept_date[i] = np.array([sch_departure.month, sch_departure.day, sch_departure.isoweekday(), sch_departure.hour * 60 + sch_departure.minute])
            delays[i] = delay

        except:
            counter += 1
    print('Number errors: ', counter)
for i, c in enumerate(carriers):
    one_hot = np.zeros(len(carriers), dtype=int)
    one_hot[i] = 1
    carriers[c] = one_hot

pprint(carriers)