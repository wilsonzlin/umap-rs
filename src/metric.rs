use ndarray::{Array1, ArrayView1};
use std::fmt::Debug;

/// A distance metric for embedding spaces.
///
/// Metrics must be able to compute both distances and their gradients for use
/// in gradient descent optimization. Thread-safety (Send + Sync) is required
/// for parallel optimization.
pub trait Metric: Debug + Send + Sync {
    /// Compute the distance between two points and its gradient.
    ///
    /// # Arguments
    ///
    /// * `a` - First point
    /// * `b` - Second point
    ///
    /// # Returns
    ///
    /// A tuple of (distance, gradient) where:
    /// - `distance` is the scalar distance value
    /// - `gradient` is ∂distance/∂a (gradient with respect to the first point)
    fn distance(&self, a: ArrayView1<f32>, b: ArrayView1<f32>) -> (f32, Array1<f32>);

    /// Distance threshold beyond which points are considered disconnected.
    ///
    /// For unbounded metrics like Euclidean, this is typically infinity.
    /// For bounded metrics (e.g., on spheres or in hyperbolic space), this
    /// should be the maximum meaningful distance.
    ///
    /// Default: f32::INFINITY
    fn disconnection_threshold(&self) -> f32 {
        f32::INFINITY
    }

    /// Optional fast path for computing squared distance without square root.
    ///
    /// For Euclidean and related metrics, computing squared distance is faster
    /// than the full distance (avoids sqrt). Return `Some(dist_squared)` if
    /// your metric supports this optimization, `None` otherwise.
    ///
    /// This is used to select between optimized (Euclidean) and generic
    /// optimization paths.
    ///
    /// Default: None
    fn squared_distance(&self, _a: ArrayView1<f32>, _b: ArrayView1<f32>) -> Option<f32> {
        None
    }
}
