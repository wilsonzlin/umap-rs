import locale
from warnings import warn

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
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


def flatten_iter(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten_iter(i):
                yield j
        else:
            yield i


def flattened(container):
    return tuple(flatten_iter(container))


def breadth_first_search(adjmat, start, min_vertices):
    explored = []
    queue = [start]
    levels = {}
    levels[start] = 0
    max_level = np.inf
    visited = [start]

    while queue:
        node = queue.pop(0)
        explored.append(node)
        if max_level == np.inf and len(explored) > min_vertices:
            max_level = max(levels.values())

        if levels[node] + 1 < max_level:
            neighbors = adjmat[node].indices
            for neighbour in neighbors:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

                    levels[neighbour] = levels[node] + 1

    return np.array(explored)


def raise_disconnected_warning(
    edges_removed,
    vertices_disconnected,
    disconnection_distance,
    total_rows,
    threshold=0.1,
    verbose=False,
):
    """A simple wrapper function to avoid large amounts of code repetition."""
    if verbose & (vertices_disconnected == 0) & (edges_removed > 0):
        print(
            f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.  "
            f"This is not a problem as no vertices were disconnected."
        )
    elif (vertices_disconnected > 0) & (
        vertices_disconnected <= threshold * total_rows
    ):
        warn(
            f"A few of your vertices were disconnected from the manifold.  This shouldn't cause problems.\n"
            f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
            f"It has only fully disconnected {vertices_disconnected} vertices.\n"
            f"Use umap.utils.disconnected_vertices() to identify them.",
        )
    elif vertices_disconnected > threshold * total_rows:
        warn(
            f"A large number of your vertices were disconnected from the manifold.\n"
            f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
            f"It has fully disconnected {vertices_disconnected} vertices.\n"
            f"You might consider using find_disconnected_points() to find and remove these points from your data.\n"
            f"Use umap.utils.disconnected_vertices() to identify them.",
        )


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each sample. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    return_dists=False,
    bipartite=False,
):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    return_dists: bool (optional, default False)
        Whether to return the pairwise distance associated with each edge.

    bipartite: bool (optional, default False)
        Does the nearest neighbour set represent a bipartite graph? That is, are the
        nearest neighbour indices from the same point set as the row indices?

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)

    dists: array of shape (n_samples * n_neighbors)
        Distance associated with each entry in the resulting sparse matrix
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
    return_dists=None,
):
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
    )

    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists
    )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos
    else:
        if return_dists:
            dmat = scipy.sparse.coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, dists


@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return


@numba.njit()
def fast_metric_intersection(
    rows, cols, values, discrete_space, metric, metric_args, scale
):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array of shape
        An array of the values of each non-zero in the sparse matrix
        representation.

    discrete_space: array of shape (n_samples, n_features)
        The vectors of categorical labels to use in the intersection.

    metric: numba function
        The function used to calculate distance over the target array.

    scale: float
        A scaling to apply to the metric.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        dist = metric(discrete_space[i], discrete_space[j], *metric_args)
        values[nz] *= np.exp(-(scale * dist))

    return


@numba.njit()
def reprocess_row(probabilities, k=15, n_iters=32):
    target = np.log2(k)

    lo = 0.0
    hi = NPY_INFINITY
    mid = 1.0

    for n in range(n_iters):

        psum = 0.0
        for j in range(probabilities.shape[0]):
            psum += pow(probabilities[j], mid)

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum < target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    return np.power(probabilities, mid)


@numba.njit()
def reset_local_metrics(simplicial_set_indptr, simplicial_set_data):
    for i in range(simplicial_set_indptr.shape[0] - 1):
        simplicial_set_data[simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]] = (
            reprocess_row(
                simplicial_set_data[
                    simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]
                ]
            )
        )
    return


def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.

    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.

    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        simplicial_set = simplicial_set.tocsr()
        reset_local_metrics(simplicial_set.indptr, simplicial_set.data)
        simplicial_set = simplicial_set.tocoo()
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


def discrete_metric_simplicial_set_intersection(
    simplicial_set,
    discrete_space,
    unknown_dist=1.0,
    far_dist=5.0,
    metric=None,
    metric_kws={},
    metric_scale=1.0,
):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from discrete metric data using discrete distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.

    TODO: optional category cardinality based weighting of distance

    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.

    discrete_space: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist: float (optional, default 5.0)
        The distance between unmatched labels.

    metric: str (optional, default None)
        If not None, then use this metric to determine the
        distance between values.

    metric_scale: float (optional, default 1.0)
        If using a custom metric scale the distance values by
        this value -- this controls the weighting of the
        intersection. Larger values weight more toward target.

    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    if metric is not None:
        # We presume target is now a 2d array, with each row being a
        # vector of target info
        if metric in dist.named_distances:
            metric_func = dist.named_distances[metric]
        else:
            raise ValueError("Discrete intersection metric is not recognized")

        fast_metric_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            metric_func,
            tuple(metric_kws.values()),
            metric_scale,
        )
    else:
        fast_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            unknown_dist,
            far_dist,
        )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


def general_simplicial_set_intersection(
    simplicial_set1, simplicial_set2, weight=0.5, right_complement=False
):

    if right_complement:
        result = simplicial_set1.tocoo()
    else:
        result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        mix_weight=weight,
        right_complement=right_complement,
    )

    return result


def general_simplicial_set_union(simplicial_set1, simplicial_set2):
    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_union(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
    )

    return result


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


# scale coords so that the largest coordinate is max_coords, then add normal-distributed
# noise with standard deviation noise
def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(
        np.float32
    )


def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    densmap,
    densmap_kwds,
    output_dens,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    # Use more epochs for densMAP
    if densmap:
        default_epochs += 200

    if n_epochs is None:
        n_epochs = default_epochs

    # If n_epoch is a list, get the maximum epoch to reach
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    init_data = np.array(init)
    if len(init_data.shape) == 2:
        if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
            tree = KDTree(init_data)
            dist, ind = tree.query(init_data, k=2)
            nndist = np.mean(dist[:, 1])
            embedding = init_data + random_state.normal(
                scale=0.001 * nndist, size=init_data.shape
            ).astype(np.float32)
        else:
            embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            tqdm_kwds=tqdm_kwds,
            move_other=True,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            output_metric,
            tuple(output_metric_kwds.values()),
            verbose=verbose,
            tqdm_kwds=tqdm_kwds,
            move_other=True,
        )

    if isinstance(embedding, list):
        aux_data["embedding_list"] = embedding
        embedding = embedding[-1].copy()

    return embedding, aux_data


@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).

    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample

    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def init_graph_transform(graph, embedding):
    """Given a bipartite graph representing the 1-simplices and strengths between the
    new points and the original data set along with an embedding of the original points
    initialize the positions of new points relative to the strengths (of their neighbors in the source data).

    If a point is in our original data set it embeds at the original points coordinates.
    If a point has no neighbours in our original dataset it embeds as the np.nan vector.
    Otherwise a point is the weighted average of it's neighbours embedding locations.

    Parameters
    ----------
    graph: csr_matrix (n_new_samples, n_samples)
        A matrix indicating the 1-simplices and their associated strengths.  These strengths should
        be values between zero and one and not normalized.  One indicating that the new point was identical
        to one of our original points.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((graph.shape[0], embedding.shape[1]), dtype=np.float32)

    for row_index in range(graph.shape[0]):
        graph_row = graph[row_index]
        if graph_row.nnz == 0:
            result[row_index] = np.nan
            continue
        row_sum = graph_row.sum()
        for graph_value, col_index in zip(graph_row.data, graph_row.indices):
            if graph_value == 1:
                result[row_index, :] = embedding[col_index, :]
                break
            result[row_index] += graph_value / row_sum * embedding[col_index]

    return result


@numba.njit()
def init_update(current_init, n_original_samples, indices):
    for i in range(n_original_samples, indices.shape[0]):
        n = 0
        for j in range(indices.shape[1]):
            for d in range(current_init.shape[1]):
                if indices[i, j] < n_original_samples:
                    n += 1
                    current_init[i, d] += current_init[indices[i, j], d]
        for d in range(current_init.shape[1]):
            current_init[i, d] /= n

    return


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAP(BaseEstimator, ClassNamePrefixFeaturesOutMixin):
    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        densmap=False,
        dens_lambda=2.0,
        dens_frac=0.3,
        dens_var_shift=0.1,
        disconnection_distance=None,
        precomputed_knn=(None, None, None),
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm

        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.disconnection_distance = disconnection_distance
        self.precomputed_knn = precomputed_knn

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in (
            "pca",
            "spectral",
            "random",
            "tswspectral",
        ):
            raise ValueError(
                'string init values must be one of: "pca", "tswspectral",'
                ' "spectral" or "random"'
            )
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        self.n_epochs_list = None
        if (
            isinstance(self.n_epochs, list)
            or isinstance(self.n_epochs, tuple)
            or isinstance(self.n_epochs, np.ndarray)
        ):
            if not issubclass(
                np.array(self.n_epochs).dtype.type, np.integer
            ) or not np.all(np.array(self.n_epochs) >= 0):
                raise ValueError(
                    "n_epochs must be a nonnegative integer "
                    "or a list of nonnegative integers"
                )
            self.n_epochs_list = list(self.n_epochs)
        elif self.n_epochs is not None and (
            self.n_epochs < 0 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError(
                "n_epochs must be a nonnegative integer "
                "or a list of nonnegative integers"
            )
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method, define a distance function that returns a tuple "
                    "of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn("using precomputed metric; inverse_transform will be unavailable")
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in dist.named_distances:
            if self._sparse_data:
                if self.metric in sparse.sparse_named_distances:
                    self._input_distance_func = sparse.sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = dist.named_distances[self.metric]
            try:
                self._inverse_distance_func = dist.named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        elif self.metric in pynn_named_distances:
            if self._sparse_data:
                if self.metric in pynn_sparse_named_distances:
                    self._input_distance_func = pynn_sparse_named_distances[self.metric]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = pynn_named_distances[self.metric]

            warn(
                "gradient function is not yet implemented for {} distance metric; "
                "inverse_transform will be unavailable".format(self.metric)
            )
            self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set output distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in dist.named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be a postive integer, or -1 (for all cores)")
        if self.n_jobs != 1 and self.random_state is not None:
            self.n_jobs = 1
            warn(
                f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism."
            )

        if self.dens_lambda < 0.0:
            raise ValueError("dens_lambda cannot be negative")
        if self.dens_frac < 0.0 or self.dens_frac > 1.0:
            raise ValueError("dens_frac must be between 0.0 and 1.0")
        if self.dens_var_shift < 0.0:
            raise ValueError("dens_var_shift cannot be negative")

        self._densmap_kwds = {
            "lambda": self.dens_lambda if self.densmap else 0.0,
            "frac": self.dens_frac if self.densmap else 0.0,
            "var_shift": self.dens_var_shift,
            "n_neighbors": self.n_neighbors,
        }

        if self.densmap:
            if self.output_metric not in ("euclidean", "l2"):
                raise ValueError(
                    "Non-Euclidean output metric not supported for densMAP."
                )

        # This will be used to prune all edges of greater than a fixed value from our knn graph.
        # We have preset defaults described in DISCONNECTION_DISTANCES for our bounded measures.
        # Otherwise a user can pass in their own value.
        if self.disconnection_distance is None:
            self._disconnection_distance = DISCONNECTION_DISTANCES.get(
                self.metric, np.inf
            )
        elif isinstance(self.disconnection_distance, int) or isinstance(
            self.disconnection_distance, float
        ):
            self._disconnection_distance = self.disconnection_distance
        else:
            raise ValueError("disconnection_distance must either be None or a numeric.")

        if self.tqdm_kwds is None:
            self.tqdm_kwds = {}
        else:
            if isinstance(self.tqdm_kwds, dict) is False:
                raise ValueError(
                    "tqdm_kwds must be a dictionary. Please provide valid tqdm "
                    "parameters as key value pairs. Valid tqdm parameters can be "
                    "found here: https://github.com/tqdm/tqdm#parameters"
                )
        if "desc" not in self.tqdm_kwds:
            self.tqdm_kwds["desc"] = "Epochs completed"
        if "bar_format" not in self.tqdm_kwds:
            bar_f = "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]"
            self.tqdm_kwds["bar_format"] = bar_f

        if hasattr(self, "knn_dists") and self.knn_dists is not None:
            if self.unique:
                raise ValueError(
                    "unique is not currently available for " "precomputed_knn."
                )
            if not isinstance(self.knn_indices, np.ndarray):
                raise ValueError("precomputed_knn[0] must be ndarray object.")
            if not isinstance(self.knn_dists, np.ndarray):
                raise ValueError("precomputed_knn[1] must be ndarray object.")
            if self.knn_dists.shape != self.knn_indices.shape:
                raise ValueError(
                    "precomputed_knn[0] and precomputed_knn[1]"
                    " must be numpy arrays of the same size."
                )
            # #848: warn but proceed if no search index is present
            if not isinstance(self.knn_search_index, NNDescent):
                warn(
                    "precomputed_knn[2] (knn_search_index) "
                    "is not an NNDescent object: transforming new data with transform "
                    "will be unavailable."
                )
            if self.knn_dists.shape[1] < self.n_neighbors:
                warn(
                    "precomputed_knn has a lower number of neighbors than "
                    "n_neighbors parameter. precomputed_knn will be ignored"
                    " and the k-nn will be computed normally."
                )
                self.knn_indices = None
                self.knn_dists = None
                self.knn_search_index = None
            elif self.knn_dists.shape[0] != self._raw_data.shape[0]:
                warn(
                    "precomputed_knn has a different number of samples than the"
                    " data you are fitting. precomputed_knn will be ignored and"
                    "the k-nn will be computed normally."
                )
                self.knn_indices = None
                self.knn_dists = None
                self.knn_search_index = None
            elif (
                self.knn_dists.shape[0] < 4096
                and not self.force_approximation_algorithm
            ):
                # force_approximation_algorithm is irrelevant for pre-computed knn
                # always set it to True which keeps downstream code paths working
                self.force_approximation_algorithm = True
            elif self.knn_dists.shape[1] > self.n_neighbors:
                # if k for precomputed_knn larger than n_neighbors we simply prune it
                self.knn_indices = self.knn_indices[:, : self.n_neighbors]
                self.knn_dists = self.knn_dists[:, : self.n_neighbors]

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))

        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    def fit(self, X, **kwargs):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        **kwargs : optional
            Any additional keyword arguments are passed to _fit_embed_data.
        """
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        init = self.init

        self._initial_alpha = self.learning_rate

        self.knn_indices = self.precomputed_knn[0]
        self.knn_dists = self.precomputed_knn[1]
        self.knn_search_index = None

        self._validate_parameters()

        # If we aren't asking for unique use the full index.
        # This will save special cases later.
        index = list(range(X.shape[0]))
        inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        assert X[index].shape[0] > self.n_neighbors
        self._n_neighbors = self.n_neighbors

        random_state = check_random_state(self.random_state)

        # Standard case
        self._small_data = False
        # Standard case
        nn_metric = self._input_distance_func
        self._knn_indices = self.knn_indices
        self._knn_dists = self.knn_dists
        self._knn_search_index = self.knn_search_index
        # Disconnect any vertices farther apart than _disconnection_distance
        disconnected_index = self._knn_dists >= self._disconnection_distance
        self._knn_indices[disconnected_index] = -1
        self._knn_dists[disconnected_index] = np.inf
        edges_removed = disconnected_index.sum()

        (
            self.graph_,
            self._sigmas,
            self._rhos,
            self.graph_dists_,
        ) = fuzzy_simplicial_set(
            X[index],
            self.n_neighbors,
            random_state,
            nn_metric,
            self._metric_kwds,
            self._knn_indices,
            self._knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            self.verbose,
            self.densmap or self.output_dens,
        )
        # Report the number of vertices with degree 0 in our umap.graph_
        # This ensures that they were properly disconnected.
        vertices_disconnected = np.sum(
            np.array(self.graph_.sum(axis=1)).flatten() == 0
        )
        raise_disconnected_warning(
            edges_removed,
            vertices_disconnected,
            self._disconnection_distance,
            self._raw_data.shape[0],
            verbose=self.verbose,
        )

        self._supervised = False

        if self.densmap or self.output_dens:
            self._densmap_kwds["graph_dists"] = self.graph_dists_

        epochs = (
            self.n_epochs_list if self.n_epochs_list is not None else self.n_epochs
        )
        self.embedding_, aux_data = self._fit_embed_data(
            self._raw_data[index],
            epochs,
            init,
            random_state,  # JH why raw data?
            **kwargs,
        )

        if self.n_epochs_list is not None:
            if "embedding_list" not in aux_data:
                raise KeyError(
                    "No list of embedding were found in 'aux_data'. "
                    "It is likely the layout optimization function "
                    "doesn't support the list of int for 'n_epochs'."
                )
            else:
                self.embedding_list_ = [
                    e[inverse] for e in aux_data["embedding_list"]
                ]

        # Assign any points that are fully disconnected from our manifold(s) to have embedding
        # coordinates of np.nan.  These will be filtered by our plotting functions automatically.
        # They also prevent users from being deceived a distance query to one of these points.
        # Might be worth moving this into simplicial_set_embedding or _fit_embed_data
        disconnected_vertices = np.array(self.graph_.sum(axis=1)).flatten() == 0
        if len(disconnected_vertices) > 0:
            self.embedding_[disconnected_vertices] = np.full(
                self.n_components, np.nan
            )

        self.embedding_ = self.embedding_[inverse]
        if self.output_dens:
            self.rad_orig_ = aux_data["rad_orig"][inverse]
            self.rad_emb_ = aux_data["rad_emb"][inverse]

        self._input_hash = joblib.hash(self._raw_data)

        # Set number of features out for sklearn API
        self._n_features_out = self.embedding_.shape[1]

        return self

    def _fit_embed_data(self, X, n_epochs, init, random_state, **kwargs):
        """A method wrapper for simplicial_set_embedding that can be
        replaced by subclasses. Arbitrary keyword arguments can be passed
        through .fit() and .fit_transform().
        """
        return simplicial_set_embedding(
            X,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self.densmap,
            self._densmap_kwds,
            self.output_dens,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
            tqdm_kwds=self.tqdm_kwds,
        )

    def fit_transform(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.embedding_

    def transform(self, X, ensure_all_finite=True):
        # If we fit just a single instance then error
        if self._raw_data.shape[0] == 1:
            raise ValueError(
                "Transform unavailable when model was fit with only a single data sample."
            )
        x_hash = joblib.hash(X)
        if x_hash == self._input_hash:
            return self.embedding_

        # #848: knn_search_index is allowed to be None if not transforming new data,
        # so now we must validate that if it exists it is not None
        if hasattr(self, "_knn_search_index") and self._knn_search_index is None:
            raise NotImplementedError(
                "No search index available: transforming data"
                " into an existing embedding is not supported"
            )

        # X = check_array(X, dtype=np.float32, order="C", accept_sparse="csr")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12
        indices, dists = self._knn_search_index.query(
            X, self.n_neighbors, epsilon=epsilon
        )

        dists = dists.astype(np.float32, order="C")
        # Remove any nearest neighbours who's distances are greater than our disconnection_distance
        indices[dists >= self._disconnection_distance] = -1
        adjusted_local_connectivity = max(0.0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist(
            dists,
            float(self._n_neighbors),
            local_connectivity=float(adjusted_local_connectivity),
        )

        rows, cols, vals, dists = compute_membership_strengths(
            indices, dists, sigmas, rhos, bipartite=True
        )

        graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # This was a very specially constructed graph with constant degree.
        # That lets us do fancy unpacking by reshaping the csr matrix indices
        # and data. Doing so relies on the constant degree assumption!
        # csr_graph = normalize(graph.tocsr(), norm="l1")
        # inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
        # weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
        # embedding = init_transform(inds, weights, self.embedding_)
        # This is less fast code than the above numba.jit'd code.
        # It handles the fact that our nearest neighbour graph can now contain variable numbers of vertices.
        csr_graph = graph.tocsr()
        csr_graph.eliminate_zeros()
        embedding = init_graph_transform(csr_graph, self.embedding_)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = int(self.n_epochs // 3.0)

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        if self.output_metric == "euclidean":
            embedding = optimize_layout_euclidean(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  # Fixes #179 & #217,
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self.random_state is None,
                verbose=self.verbose,
                tqdm_kwds=self.tqdm_kwds,
            )
        else:
            embedding = optimize_layout_generic(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  # Fixes #179 & #217
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self._output_distance_func,
                tuple(self._output_metric_kwds.values()),
                verbose=self.verbose,
                tqdm_kwds=self.tqdm_kwds,
            )

        return embedding
