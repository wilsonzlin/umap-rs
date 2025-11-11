use crate::umap::compute_membership_strengths::ComputeMembershipStrengths;
use crate::umap::smooth_knn_dist::SmoothKnnDist;
use dashmap::DashSet;
use itertools::izip;
use ndarray::Array1;
use ndarray::ArrayView2;
use sprs::CsMat;
use sprs::CsMatView;
use sprs::TriMat;
use std::iter::zip;
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

    let mut result = {
      let mut mat = TriMat::new((n_samples, n_samples));
      for (v, r, c) in izip!(vals, rows, cols) {
        if v != 0.0 {
          mat.add_triplet(r as usize, c as usize, v);
        }
      }
      mat.to_csr::<usize>()
    };

    if apply_set_operations {
      let transpose = result.transpose_view().to_csr();
      let prod_matrix = hadamard(&result.view(), &transpose.view());

      // Compute: set_op_mix_ratio * (result + transpose - prod_matrix) + (1 - set_op_mix_ratio) * prod_matrix
      // This simplifies to: set_op_mix_ratio * (result + transpose) + (1 - 2*set_op_mix_ratio) * prod_matrix
      let mut tri = TriMat::new(result.shape());

      // Add set_op_mix_ratio * (result + transpose)
      for (val, (row, col)) in result.iter() {
        tri.add_triplet(row, col, set_op_mix_ratio * val);
      }
      for (val, (row, col)) in transpose.iter() {
        tri.add_triplet(row, col, set_op_mix_ratio * val);
      }

      // Add (1 - 2*set_op_mix_ratio) * prod_matrix
      let prod_coeff = 1.0 - 2.0 * set_op_mix_ratio + set_op_mix_ratio;
      for (val, (row, col)) in prod_matrix.iter() {
        tri.add_triplet(row, col, prod_coeff * val);
      }

      result = tri.to_csr::<usize>();
    }

    (result, sigmas, rhos)
  }
}

/// Compute elementwise (Hadamard) product of two same‑shape CSRs.
fn hadamard(a: &CsMatView<f32>, b: &CsMatView<f32>) -> CsMat<f32> {
  assert_eq!(a.shape(), b.shape(), "shapes must match for hadamard");
  let mut tri = TriMat::new(a.shape());
  // iterate nonzeros of `a`
  for (row, vec) in a.outer_iterator().enumerate() {
    for (&col, &av) in zip(vec.indices().iter(), vec.data()) {
      if let Some(&bv) = b.get(row, col) {
        let prod = av * bv;
        if prod != 0.0 {
          tri.add_triplet(row, col, prod);
        }
      }
    }
  }
  tri.to_csr::<usize>()
}
