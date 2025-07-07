import locale
from warnings import warn

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import numba

from umap.layouts import (
    optimize_layout_euclidean,
    optimize_layout_generic,
)

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

DISCONNECTION_DISTANCES = {
    "correlation": 2,
    "cosine": 2,
    "hellinger": 1,
    "jaccard": 1,
    "bit_jaccard": 1,
    "dice": 1,
}
