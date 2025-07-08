use ndarray::{Array1, ArrayView1};
use std::fmt::Debug;

pub trait Metric: Debug {
  /// Calculates the distance and gradient.
  fn distance(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> (f32, Array1<f32>);
  fn is_euclidean(&self) -> bool;
  // Corresponds to DISCONNECTION_DISTANCES in umap_.py.
  fn default_disconnection_distance(&self) -> f32;
}
