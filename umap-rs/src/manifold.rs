use crate::umap::SparseMat;
use ndarray::Array1;
use serde::Deserialize;
use serde::Serialize;

/// A learned manifold representation from high-dimensional data.
///
/// This captures the fuzzy topological structure (graph) and local geometry
/// (sigmas, rhos) learned from the input data via k-nearest neighbors.
/// This is the expensive part of UMAP that can be cached and reused.
///
/// The manifold is independent of the embedding dimensionality and can be
/// used to create multiple different embeddings or continue optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedManifold {
  /// Fuzzy simplicial set as a sparse matrix with u32 indices.
  /// graph[i,j] represents the membership strength of edge i->j.
  pub(crate) graph: SparseMat,

  /// Normalization factors from local manifold approximation.
  /// sigma[i] is the distance to the local_connectivity'th nearest neighbor.
  pub(crate) sigmas: Array1<f32>,

  /// Distance to nearest neighbor for each point.
  /// Used for transform operations (future work).
  pub(crate) rhos: Array1<f32>,

  /// Number of vertices (samples) in the manifold.
  pub(crate) n_vertices: usize,

  /// Parameter 'a' of the distance-to-probability curve: 1 / (1 + a*x^(2b))
  /// Derived from min_dist and spread during manifold learning.
  pub(crate) a: f32,

  /// Parameter 'b' of the distance-to-probability curve: 1 / (1 + a*x^(2b))
  /// Derived from min_dist and spread during manifold learning.
  pub(crate) b: f32,
}

impl LearnedManifold {
  /// Get the number of vertices in the manifold.
  pub fn n_vertices(&self) -> usize {
    self.n_vertices
  }

  /// Get the curve parameters (a, b).
  pub fn curve_params(&self) -> (f32, f32) {
    (self.a, self.b)
  }

  /// Get a reference to the graph structure.
  pub fn graph(&self) -> &SparseMat {
    &self.graph
  }
}
