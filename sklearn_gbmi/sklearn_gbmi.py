from __future__ import division


__all__ = ['h', 'h_all_pairs']


import itertools

import math

import numpy as np

import sklearn.inspection.partial_dependence as partial_dependence


def h(gbm, array_or_frame, indices_or_columns = 'all'):
    """
    PURPOSE

    Compute Friedman and Popescu's H statistic, in order to look for an interaction in the passed gradient-boosting
    model among the variables represented by the elements of the passed array or frame and specified by the passed
    indices or columns.

    See Jerome H. Friedman and Bogdan E. Popescu, 2008, "Predictive learning via rule ensembles", Ann. Appl. Stat.
    2:916-954, http://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908046, s. 8.1.


    ARGUMENTS

    gbm should be a scikit-learn gradient-boosting model (instance of sklearn.ensemble.GradientBoostingClassifier or
    sklearn.ensemble.GradientBoostingRegressor) that has been fitted to array_or_frame (and a target, not used here).

    array_or_frame should be a two-dimensional NumPy array or a pandas data frame (instance of numpy.ndarray or pandas
    .DataFrame).

    indices_or_columns is optional, with default value 'all'. It should be 'all' or a list of indices of columns of
    array_or_frame if array_or_frame is a NumPy array or a list of columns of array_or_frame if array_or_frame is a
    pandas data frame. If it is 'all', then all columns of array_or_frame are used.


    RETURNS

    The H statistic of the variables or NaN if the computation is spoiled by weak main effects and rounding errors.

    H varies from 0 to 1. The larger H, the stronger the evidence for an interaction among the variables.


    EXAMPLES

    Friedman and Popescu's (2008) formulas (44) and (46) correspond to

        h(F, x, [j, k])

    and

        h(F, x, [j, k, l])

    respectively.


    NOTES

    1. Per Friedman and Popescu, only variables with strong main effects should be examined for interactions. Strengths 
    of main effects are available as gbm.feature_importances_ once gbm has been fitted.

    2. Per Friedman and Popescu, collinearity among variables can lead to interactions in gbm that are not present in
    the target function. To forestall such spurious interactions, check for strong correlations among variables before
    fitting gbm.
    """

    if indices_or_columns == 'all':
        if gbm.max_depth < array_or_frame.shape[1]:
            raise \
                Exception(
                    "gbm.max_depth == {} < array_or_frame.shape[1] == {}, so indices_or_columns must not be 'all'."
                    .format(gbm.max_depth, array_or_frame.shape[1])
                )
    else:
        if gbm.max_depth < len(indices_or_columns):
            raise \
                Exception(
                    "gbm.max_depth == {}, so indices_or_columns must contain at most {} {}."
                    .format(gbm.max_depth, gbm.max_depth, "element" if gbm.max_depth == 1 else "elements")
                )
    check_args_contd(array_or_frame, indices_or_columns)

    arr, model_inds = get_arr_and_model_inds(array_or_frame, indices_or_columns)

    width = arr.shape[1]
    f_vals = {}
    for n in range(width, 0, -1):
        for inds in itertools.combinations(range(width), n):
            f_vals[inds] = compute_f_vals(gbm, model_inds, arr, inds)

    return compute_h_val(f_vals, arr, tuple(range(width)))


def h_all_pairs(gbm, array_or_frame, indices_or_columns = 'all'):
    """
    PURPOSE

    Compute Friedman and Popescu's two-variable H statistic, in order to look for an interaction in the passed gradient-
    boosting model between each pair of variables represented by the elements of the passed array or frame and specified
    by the passed indices or columns.

    See Jerome H. Friedman and Bogdan E. Popescu, 2008, "Predictive learning via rule ensembles", Ann. Appl. Stat.
    2:916-954, http://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908046, s. 8.1.


    ARGUMENTS

    gbm should be a scikit-learn gradient-boosting model (instance of sklearn.ensemble.GradientBoostingClassifier or
    sklearn.ensemble.GradientBoostingRegressor) that has been fitted to array_or_frame (and a target, not used here).

    array_or_frame should be a two-dimensional NumPy array or a pandas data frame (instance of numpy.ndarray or pandas
    .DataFrame).

    indices_or_columns is optional, with default value 'all'. It should be 'all' or a list of indices of columns of
    array_or_frame if array_or_frame is a NumPy array or a list of columns of array_or_frame if array_or_frame is a
    pandas data frame. If it is 'all', then all columns of array_or_frame are used.


    RETURNS

    A dict whose keys are pairs (2-tuples) of indices or columns and whose values are the H statistic of the pairs of
    variables or NaN if a computation is spoiled by weak main effects and rounding errors.

    H varies from 0 to 1. The larger H, the stronger the evidence for an interaction between a pair of variables.


    EXAMPLE

    Friedman and Popescu's (2008) formula (44) for every j and k corresponds to

        h_all_pairs(F, x)


    NOTES

    1. Per Friedman and Popescu, only variables with strong main effects should be examined for interactions. Strengths 
    of main effects are available as gbm.feature_importances_ once gbm has been fitted.

    2. Per Friedman and Popescu, collinearity among variables can lead to interactions in gbm that are not present in
    the target function. To forestall such spurious interactions, check for strong correlations among variables before
    fitting gbm.
    """

    if gbm.max_depth < 2:
        raise Exception("gbm.max_depth must be at least 2.")
    check_args_contd(array_or_frame, indices_or_columns)

    arr, model_inds = get_arr_and_model_inds(array_or_frame, indices_or_columns)

    width = arr.shape[1]
    f_vals = {}
    for n in [2, 1]:
        for inds in itertools.combinations(range(width), n):
            f_vals[inds] = compute_f_vals(gbm, model_inds, arr, inds)

    h_vals = {}
    for inds in itertools.combinations(range(width), 2):
        h_vals[inds] = compute_h_val(f_vals, arr, inds)
    if indices_or_columns != 'all':
        h_vals = {tuple(model_inds[(inds,)]): h_vals[inds] for inds in h_vals.keys()}
    if not isinstance(array_or_frame, np.ndarray):
        all_cols = array_or_frame.columns.values
        h_vals = {tuple(all_cols[(inds,)]): h_vals[inds] for inds in h_vals.keys()}

    return h_vals


def check_args_contd(array_or_frame, indices_or_columns):
    if indices_or_columns != 'all':
        if len(indices_or_columns) < 2:
            raise Exception("indices_or_columns must be 'all' or contain at least 2 elements.")
        if isinstance(array_or_frame, np.ndarray):
            all_inds = range(array_or_frame.shape[1])
            if not all(ind in all_inds for ind in indices_or_columns):
                raise Exception("indices_or_columns must be 'all' or a subset of {}.".format(all_inds))
        else:
            all_cols = array_or_frame.columns.tolist()
            if not all(col in all_cols for col in indices_or_columns):
                raise Exception("indices_or_columns must be 'all' or a subset of {}.".format(all_cols))


def get_arr_and_model_inds(array_or_frame, indices_or_columns):
    if isinstance(array_or_frame, np.ndarray):
        if indices_or_columns == 'all': indices_or_columns = range(array_or_frame.shape[1])
        arr = array_or_frame[:, indices_or_columns]
        model_inds = np.array(indices_or_columns)
    else:
        all_cols = array_or_frame.columns.tolist()
        if indices_or_columns == 'all': indices_or_columns = all_cols
        arr = array_or_frame[indices_or_columns].values
        model_inds = np.array([all_cols.index(col) for col in indices_or_columns])
    return arr, model_inds


def compute_f_vals(gbm, model_inds, arr, inds):
    feat_vals, feat_val_counts = unique_rows_with_counts(arr[:, inds])
    uncentd_f_vals = partial_dependence.partial_dependence(gbm, model_inds[(inds,)], grid = feat_vals)[0][0]
    mean_uncentd_f_val = np.dot(feat_val_counts, uncentd_f_vals)/arr.shape[0]
    f_vals = uncentd_f_vals-mean_uncentd_f_val
    return dict(zip(map(tuple, feat_vals), f_vals))


def compute_h_val(f_vals, arr, inds):
    feat_vals, feat_val_counts = unique_rows_with_counts(arr)
    uniq_height = feat_vals.shape[0]
    numer_els = np.zeros(uniq_height)
    denom_els = np.empty_like(numer_els)
    for i in range(uniq_height):
        feat_vals_i = feat_vals[i]
        sign = 1.0
        for n in range(len(inds), 0, -1):
            for subinds in itertools.combinations(inds, n):
                numer_els[i] += sign*f_vals[subinds][tuple(feat_vals_i[(subinds,)])]
            sign *= -1.0
        denom_els[i] = f_vals[inds][tuple(feat_vals_i[(inds,)])]
    numer = np.dot(feat_val_counts, numer_els**2)
    denom = np.dot(feat_val_counts, denom_els**2)
    return math.sqrt(numer/denom) if numer < denom else np.nan


def unique_rows_with_counts(inp_arr):
    width = inp_arr.shape[1]
    cont_arr = np.ascontiguousarray(inp_arr)
    tuple_dtype = [(str(i), inp_arr.dtype) for i in range(width)]
    tuple_arr = cont_arr.view(tuple_dtype)
    uniq_arr, counts = np.unique(tuple_arr, return_counts = True)
    outp_arr = uniq_arr.view(inp_arr.dtype).reshape(-1, width)
    return outp_arr, counts
