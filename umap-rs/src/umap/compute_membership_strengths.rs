use dashmap::DashSet;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rayon::prelude::*;
use typed_builder::TypedBuilder;

/*
  Construct the membership strength data for the 1-skeleton of each local
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
*/
#[derive(TypedBuilder)]
pub struct ComputeMembershipStrengths<'a, 's, 'r, 'd> {
  knn_indices: ArrayView2<'a, u32>,
  knn_dists: ArrayView2<'a, f32>,
  // [DIVERGE] Instead of overwriting knn_indices with -1 for disconnections, which requires mutating or expensive copying, we instead just maintain set of disconnections.
  knn_disconnections: &'d DashSet<(usize, usize)>,
  sigmas: &'s ArrayView1<'s, f32>,
  rhos: &'r ArrayView1<'r, f32>,
  #[builder(default = false)]
  bipartite: bool,
}

impl<'a, 's, 'r, 'd> ComputeMembershipStrengths<'a, 's, 'r, 'd> {
  pub fn exec(self) -> (Array1<u32>, Array1<u32>, Array1<f32>) {
    let Self {
      knn_indices,
      knn_dists,
      knn_disconnections,
      sigmas,
      rhos,
      bipartite,
    } = self;

    let n_samples = knn_indices.shape()[0];
    let n_neighbors = knn_indices.shape()[1];

    // OPTIMIZATION: Parallelize by computing each row's worth of data in parallel
    let results: Vec<_> = (0..n_samples)
      .into_par_iter()
      .flat_map(|i| {
        (0..n_neighbors)
          .filter_map(|j| {
            if knn_disconnections.contains(&(i, j)) {
              return None;
            }

            let knn_idx = knn_indices[(i, j)];

            // If applied to an adjacency matrix points shouldn't be similar to themselves.
            // If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            let val = if !bipartite && knn_idx == i as u32 {
              0.0
            } else if knn_dists[(i, j)] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
              1.0
            } else {
              // OPTIMIZATION: Compute exp directly
              f32::exp(-(knn_dists[(i, j)] - rhos[i]) / sigmas[i])
            };

            Some((i as u32, knn_idx, val))
          })
          .collect::<Vec<_>>()
      })
      .collect();

    // Unzip into separate arrays
    let (rows, cols_vals): (Vec<_>, Vec<_>) = results.iter().map(|(r, c, v)| (*r, (*c, *v))).unzip();
    let (cols, vals): (Vec<_>, Vec<_>) = cols_vals.into_iter().unzip();

    (
      Array1::from(rows),
      Array1::from(cols),
      Array1::from(vals),
    )
  }
}
