# Python implementation of the Nadeau and Bengio correction of dependent Student's t-test
# using the equation stated in https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf

from scipy.stats import t
from math import sqrt
from statistics import stdev

def corrected_dependent_ttest(data1, data2, n_training_folds, n_test_folds, alpha): #è per davvero uguale alla formula
    n = len(data1) #n è il numero di campioni
    differences = [(data1[i]-data2[i]) for i in range(n)] #xij = aij − bij for fold i and run j
    sd = stdev(differences)
    divisor = 1 / n * sum(differences) #d
    test_training_ratio = n_test_folds / n_training_folds #n1/n2
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    #calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0 #formula comune anche su google
    # return everything
    return t_stat, df, cv, p

def corrected_dependent_ttest_kfold(data1, data2, k, alpha): #formule Kuncheva
    n = len(data1) #n è il numero di campioni
    differences = [(data1[i]-data2[i]) for i in range(n)] #xij = aij − bij for fold i and run j
    sd = stdev(differences)
    divisor = 1 / n * sum(differences) #d
    denominator = sqrt((2 * k - 1) / (k * (k - 1))) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    #calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = 2 * t.cdf(-(abs(divisor)/denominator), df)
    # return everything
    return t_stat, df, cv, p