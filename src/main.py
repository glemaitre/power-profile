#!/usr/bin/env python

# Sample usage of python-fitparse to parse an activity and
# print its data records.


from fitparse import Activity

activity = Activity("/home/cedric/Documents/Code/Sport/power-profile/data/user_1/2015/2015-08-09-13-14-05.fit")
activity.parse()

# Records of type 'record' (I know, confusing) are the entries in an
# activity file that represent actual data points in your workout.
records = activity.get_records_by_type('record')
current_record_number = 0

for record in records:
    # Print record number
    current_record_number += 1
    print (" Record #%d " % current_record_number).center(40, '-')

    # Get the list of valid fields on this record
    valid_field_names = record.get_valid_field_names()

    for field_name in valid_field_names:
        # Get the data and units for the field
        field_data = record.get_data(field_name)
        field_units = record.get_units(field_name)
        
        # Print what we've got!
        if field_units:
            print " * %s: %s %s" % (field_name, field_data, field_units)
        else:
            print " * %s: %s" % (field_name, field_data)
            
    print
