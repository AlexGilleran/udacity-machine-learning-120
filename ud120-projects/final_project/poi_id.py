#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn.cross_validation import train_test_split

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    zipped = zip(ages, net_worths, predictions)
    mapped = map(lambda tuple: (tuple[0], tuple[1], tuple[2] - tuple[1]), zipped)
    sorted2 = sorted(mapped, key=lambda tuple: -tuple[2])
    newLength = int((len(sorted2) * 0.9))
    cleaned_data = sorted2[:newLength]

    ### your code goes here


    return cleaned_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_stock_value', 'restricted_stock_deferred', 'bonus', 'expenses', 'to_poi_proportion', 'shared_poi_proportion'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop( "TOTAL", 0 )
data_dict = {k: v for k, v in data_dict.iteritems() if all(elem >= 0 for elem in map(lambda feature_name: v[feature_name], ['from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi']))}
data_dict = {k: v for k, v in data_dict.iteritems() if all(elem != 'NaN' for elem in map(lambda feature_name: v[feature_name], ['from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi']))}

# for point in abc:
#     poi = point[0]
#     x = point[1]
#     y = point[2]
#     color = 'green' if  poi == 1 else 'red'
#     matplotlib.pyplot.scatter( x, y, color = color )

# matplotlib.pyplot.plot(a_train, predicted, color="blue")
# matplotlib.pyplot.scatter(abc[0], abc[1], color="green")
#
#

# matplotlib.pyplot.savefig("test3.png")



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key, value in data_dict.iteritems():
    if (value['from_poi_to_this_person'] == 'NaN'):
        value['from_poi_to_this_person'] = 0
    if (value['from_this_person_to_poi'] == 'NaN'):
        value['from_this_person_to_poi'] = 0
    if (value['shared_receipt_with_poi'] == 'NaN'):
        value['shared_receipt_with_poi'] = 0

    value['from_poi_proportion'] = float(value['from_poi_to_this_person']) / max(1, float(value['to_messages']))
    value['to_poi_proportion'] = float(value['from_this_person_to_poi']) / max(1, float(value['from_messages']))
    value['shared_poi_proportion'] = float(value['shared_receipt_with_poi']) / max(1, float(value['shared_receipt_with_poi']))
print(data_dict)
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
# clf = GaussianNB()
clf = AdaBoostClassifier(
                         DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
                         learning_rate=0.65,
                         n_estimators=50)
# clf = LinearSVC(C = 0.1)
# clf = MultinomialNB(alpha=0)
# clf = KNeighborsClassifier(3, weights='distance')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2)



clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list, folds = 50)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


