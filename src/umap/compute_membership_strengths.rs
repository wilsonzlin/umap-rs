def compute_membership_strengths(
  knn_indices,
  knn_dists,
  sigmas,
  rhos,
  bipartite=False,
):
  n_samples = knn_indices.shape[0]
  n_neighbors = knn_indices.shape[1]

  rows = np.zeros(knn_indices.size, dtype=np.int32)
  cols = np.zeros(knn_indices.size, dtype=np.int32)
  vals = np.zeros(knn_indices.size, dtype=np.float32)
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

  return rows, cols, vals, dists
