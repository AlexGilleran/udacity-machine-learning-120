#!/usr/bin/python


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
    sorted2 = sorted(mapped, key=lambda tuple: tuple[2])
    newLength = int((len(sorted2) * 0.9))
    cleaned_data = sorted2[:newLength]

    ### your code goes here

    
    return cleaned_data

