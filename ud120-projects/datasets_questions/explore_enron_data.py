#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print(len(enron_data))
pois = {k: v for k, v in enron_data.iteritems() if v['poi'] == 1}
print(len(pois))
totalPayments = {k: v for k, v in pois.iteritems() if v['total_payments'] == 'NaN'}
print(len(totalPayments))

# my_dictionary = {k: v["total_payments"] for k, v in enron_data.items()}
# for key, value in sorted(my_dictionary.iteritems(), key=lambda (k, v): (v, k)):
#     print "%s: %s" % (key, value)

# salaries = {k: v for k, v in pois.iteritems() if v['total_payments'] == 'NaN'}
# print("%s, %s" % (len(salaries), len(salaries) / float(len(pois))))
# for key, value in enron_data.iteritems():
#     print "%s: %s" % (key, value)
# print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])