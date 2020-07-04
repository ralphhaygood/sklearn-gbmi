# Adapted by Ralph Haygood from the late, lamented sklearn.ensemble.partial_dependence.partial_dependence by Peter Prettenhofer.
#
# The function sklearn.ensemble.partial_dependence.partial_dependence, on which this package (sklearn-gbmi) depended, doesn't exist anymore. It was partially
# replaced by sklearn.inspection.partial_dependence, but only partially, in that the new function doesn't accept a specified grid as the old one did; compare
#
# https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.ensemble.partial_dependence.partial_dependence/
#
# about the old function with
#
# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
#
# about the new function, and note the disappearance of the grid argument. Accordingly, I've more or less copied Peter Prettenhofer's old code into this file
# and _partial_dependence_tree.pyx, omitting parts unneeded here.

import numpy as np

from sklearn.tree._tree import DTYPE

from ._partial_dependence_tree import _partial_dependence_tree

def partial_dependence(gbrt, target_variables, grid):
    target_variables = np.asarray(target_variables, dtype=np.int32, order='C').ravel()
    grid = np.asarray(grid, dtype=DTYPE, order='C')
    n_trees_per_stage = gbrt.estimators_.shape[1]
    n_estimators = gbrt.estimators_.shape[0]
    pdp = np.zeros((n_trees_per_stage, grid.shape[0],), dtype=np.float64, order='C')
    for stage in range(n_estimators):
        for k in range(n_trees_per_stage):
            tree = gbrt.estimators_[stage, k].tree_
            _partial_dependence_tree(tree, grid, target_variables, gbrt.learning_rate, pdp[k])
    return pdp
