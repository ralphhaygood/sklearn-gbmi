"""
PURPOSE

Compute Friedman and Popescu's H statistics, in order to look for interactions among variables in scikit-learn
gradient-boosting models.

See Jerome H. Friedman and Bogdan E. Popescu, 2008, "Predictive learning via rule ensembles", Ann. Appl. Stat.
2:916-954, http://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908046, s. 8.1.


USAGE

Given a scikit-learn gradient-boosting model gbm that has been fitted to a NumPy array or pandas data frame
array_or_frame and a list of indices of columns of the array or columns of the data frame indices_or_columns, the H
statistic of the variables represented by the elements of array_or_frame and specified by indices_or_columns can be
computed via

    from sklearn_gbmi import *
    h(gbm, array_or_frame, indices_or_columns)

Alternatively, the two-variable H statistic of each pair of variables represented by the elements of array_or_frame and
specified by indices_or_columns can be computed via

    from sklearn_gbmi import *
    h_all_pairs(gbm, array_or_frame, indices_or_columns)

(Compared to iteratively calling h, calling h_all_pairs avoids redundant computations.)

indices_or_columns is optional, with default value 'all'. If it is 'all', then all columns of array_or_frame are used.

NaN is returned if a computation is spoiled by weak main effects and rounding errors.

H varies from 0 to 1. The larger H, the stronger the evidence for an interaction among the variables.


NOTES

1. Per Friedman and Popescu, only variables with strong main effects should be examined for interactions. Strengths of
main effects are available as gbm.feature_importances_ once gbm has been fitted.

2. Per Friedman and Popescu, collinearity among variables can lead to interactions in gbm that are not present in the
target function. To forestall such spurious interactions, check for strong correlations among variables before fitting
gbm.
"""

from .sklearn_gbmi import *
