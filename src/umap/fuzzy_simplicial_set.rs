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
):
  knn_dists = knn_dists.astype(np.float32)

  sigmas, rhos = smooth_knn_dist(
      knn_dists,
      float(n_neighbors),
      local_connectivity=float(local_connectivity),
  )

  rows, cols, vals, dists = compute_membership_strengths(
      knn_indices, knn_dists, sigmas, rhos
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

  dists = None

  return result, sigmas, rhos, dists
