use crate::umap::compute_membership_strengths::ComputeMembershipStrengths;
use crate::umap::smooth_knn_dist::SmoothKnnDist;
use dashmap::DashSet;
use ndarray::Array1;
use ndarray::ArrayView2;
use rayon::prelude::*;
use sprs::CsMat;
use sprs::TriMat;
use std::collections::HashSet;
use typed_builder::TypedBuilder;

/*
  Given a set of data X, a neighborhood size, and a measure of distance
  compute the fuzzy simplicial set (here represented as a fuzzy graph in
  the form of a sparse matrix) associated to the data. This is done by
  locally approximating geodesic distance at each point, creating a fuzzy
  simplicial set for each such point, and then combining all the local
  fuzzy simplicial sets into a global one via a fuzzy union.

  Parameters
  ----------
  X: array of shape (n_samples, n_features)
      The data to be modelled as a fuzzy simplicial set.

  n_neighbors: int
      The number of neighbors to use to approximate geodesic distance.
      Larger numbers induce more global estimates of the manifold that can
      miss finer detail, while smaller values will focus on fine manifold
      structure to the detriment of the larger picture.

  knn_indices: array of shape (n_samples, n_neighbors) (optional)
      If the k-nearest neighbors of each point has already been calculated
      you can pass them in here to save computation time. This should be
      an array with the indices of the k-nearest neighbors as a row for
      each data point.

  knn_dists: array of shape (n_samples, n_neighbors) (optional)
      If the k-nearest neighbors of each point has already been calculated
      you can pass them in here to save computation time. This should be
      an array with the distances of the k-nearest neighbors as a row for
      each data point.

  set_op_mix_ratio: float (optional, default 1.0)
      Interpolate between (fuzzy) union and intersection as the set operation
      used to combine local fuzzy simplicial sets to obtain a global fuzzy
      simplicial sets. Both fuzzy set operations use the product t-norm.
      The value of this parameter should be between 0.0 and 1.0; a value of
      1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
      intersection.

  local_connectivity: int (optional, default 1)
      The local connectivity required -- i.e. the number of nearest
      neighbors that should be assumed to be connected at a local level.
      The higher this value the more connected the manifold becomes
      locally. In practice this should be not more than the local intrinsic
      dimension of the manifold.

  verbose: bool (optional, default False)
      Whether to report information on the current progress of the algorithm.

  return_dists: bool or None (optional, default None)
      Whether to return the pairwise distance associated with each edge.

  Returns
  -------
  fuzzy_simplicial_set: coo_matrix
      A fuzzy simplicial set represented as a sparse matrix. The (i,
      j) entry of the matrix represents the membership strength of the
      1-simplex between the ith and jth sample points.
*/
#[derive(TypedBuilder, Debug)]
pub struct FuzzySimplicialSet<'a, 'd> {
  n_samples: usize,
  n_neighbors: usize,
  knn_indices: ArrayView2<'a, u32>,
  knn_dists: ArrayView2<'a, f32>,
  knn_disconnections: &'d DashSet<(usize, usize)>,
  #[builder(default = 1.0)]
  set_op_mix_ratio: f32,
  #[builder(default = 1.0)]
  local_connectivity: f32,
  #[builder(default = true)]
  apply_set_operations: bool,
}

impl<'a, 'd> FuzzySimplicialSet<'a, 'd> {
  pub fn exec(self) -> (CsMat<f32>, Array1<f32>, Array1<f32>) {
    // Extract the fields we need
    let knn_dists = self.knn_dists;
    let knn_indices = self.knn_indices;
    let knn_disconnections = self.knn_disconnections;
    let n_neighbors = self.n_neighbors;
    let n_samples = self.n_samples;
    let local_connectivity = self.local_connectivity;
    let set_op_mix_ratio = self.set_op_mix_ratio;
    let apply_set_operations = self.apply_set_operations;

    let (sigmas, rhos) = SmoothKnnDist::builder()
      .distances(knn_dists)
      .k(n_neighbors)
      .local_connectivity(local_connectivity)
      .build()
      .exec();

    let (rows, cols, vals) = ComputeMembershipStrengths::builder()
      .knn_indices(knn_indices)
      .knn_dists(knn_dists)
      .knn_disconnections(knn_disconnections)
      .rhos(&rhos.view())
      .sigmas(&sigmas.view())
      .build()
      .exec();

    // Parallel triplet collection: filter zero values in parallel
    let triplets: Vec<(usize, usize, f32)> = rows
      .as_slice()
      .unwrap()
      .par_iter()
      .zip(cols.as_slice().unwrap().par_iter())
      .zip(vals.as_slice().unwrap().par_iter())
      .filter_map(|((&r, &c), &v)| {
        if v != 0.0 {
          Some((r as usize, c as usize, v))
        } else {
          None
        }
      })
      .collect();

    // Build TriMat from collected triplets (sequential but fast - just appending)
    let mut mat = TriMat::with_capacity((n_samples, n_samples), triplets.len());
    for (r, c, v) in triplets {
      mat.add_triplet(r, c, v);
    }
    let mut result = mat.to_csr::<usize>();

    if apply_set_operations {
      result = apply_set_operations_parallel(&result, set_op_mix_ratio);
    }

    (result, sigmas, rhos)
  }
}

/// Apply fuzzy set union/intersection operations in a single parallel pass.
///
/// Computes: set_op_mix_ratio * (A + A^T) + (1 - 2*set_op_mix_ratio) * (A ⊙ A^T)
/// where ⊙ is the Hadamard (elementwise) product.
///
/// This fuses the transpose, hadamard, and combination into one parallel operation
/// by computing each output value directly from the formula.
fn apply_set_operations_parallel(result: &CsMat<f32>, set_op_mix_ratio: f32) -> CsMat<f32> {
  let shape = result.shape();
  let n_samples = shape.0;

  // Collect all canonical pairs (min(i,j), max(i,j)) from non-zero positions
  // This avoids processing the same pair twice
  let canonical_pairs: HashSet<(usize, usize)> = result
    .iter()
    .map(|(_, (r, c))| (r.min(c), r.max(c)))
    .collect();

  // Convert to Vec for parallel iteration
  let pairs: Vec<(usize, usize)> = canonical_pairs.into_iter().collect();

  // Parallel computation of fused formula for each position
  let triplets: Vec<(usize, usize, f32)> = pairs
    .into_par_iter()
    .flat_map(|(a, b)| {
      // Look up values: result[a,b] and result[b,a]
      let val_ab = result.get(a, b).copied().unwrap_or(0.0);
      let val_ba = result.get(b, a).copied().unwrap_or(0.0);

      // Fused formula: mix * (A + A^T) + (1 - 2*mix) * (A ⊙ A^T)
      // At position (a, b): mix * val_ab + mix * val_ba + (1 - 2*mix) * val_ab * val_ba
      let prod_coeff = 1.0 - 2.0 * set_op_mix_ratio;
      let final_val =
        set_op_mix_ratio * val_ab + set_op_mix_ratio * val_ba + prod_coeff * val_ab * val_ba;

      let mut out = Vec::with_capacity(2);

      if final_val != 0.0 {
        out.push((a, b, final_val));
      }

      // For off-diagonal elements, also emit (b, a) with same value
      // (the formula is symmetric: mix*x + mix*y + (1-2*mix)*xy = mix*y + mix*x + (1-2*mix)*yx)
      if a != b && final_val != 0.0 {
        out.push((b, a, final_val));
      }

      out
    })
    .collect();

  // Build final TriMat from computed triplets
  let mut tri = TriMat::with_capacity((n_samples, n_samples), triplets.len());
  for (r, c, v) in triplets {
    tri.add_triplet(r, c, v);
  }
  tri.to_csr::<usize>()
}
