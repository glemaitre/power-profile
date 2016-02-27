#!/usr/bin/env python

# Sample usage of python-fitparse to parse an activity and
# print its data records.


from fitparse import FitFile

activity = FitFile("../data/user_1/2015/2015-08-09-13-14-05.fit")
activity.parse()

# Records of type 'record' (I know, confusing) are the entries in an
# activity file that represent actual data points in your workout.
records = list(activity.get_messages(name='record'))
current_record_number = 0

for record in records:
    # Print record number
    current_record_number += 1
    print (" Record #%d " % current_record_number).center(40, '-')

    for field in record:
        # Get the data and units for the field
        print (field.name, field.value, field.units)

    print
