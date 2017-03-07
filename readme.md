This is everything I did for the udacity machine learning course.

I didn't do anything super-clever for the final project, just made the 
poi emails into a proportion rather than a count, and threw in the
financial indicators that pumped up the score.

Tried a number of classifiers (they're commented out). GaussianNB was
surprisingly hard to beat but in the end the Adaboost golden hammer
won out.

    AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'),
          learning_rate=0.65, n_estimators=50, random_state=None)
	Accuracy: 0.88856	Precision: 0.49855	Recall: 0.51500	F1: 0.50664	F2: 0.51162
	Total predictions: 9000	True positives:  515	False positives:  518	False negatives:  485	True negatives: 7482
	
Would've been cool to do some text prediction on the emails but I really
can't be bothered learning enough python to read a billion text files
and efficiently parse them such that they can be used for machine
learning.