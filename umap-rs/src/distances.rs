use crate::metric::Metric;
use ndarray::Array1;
use ndarray::ArrayView1;

/// Euclidean (L2) distance metric.
///
/// Computes the standard Euclidean distance: sqrt(sum((x_i - y_i)^2))
/// Provides optimized squared distance computation for faster optimization.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
  /// Compute Euclidean distance and its gradient.
  ///
  /// Returns (distance, gradient) where gradient = (x - y) / (distance + ε)
  /// The epsilon term prevents division by zero when points are identical.
  fn distance(&self, x: ArrayView1<f32>, y: ArrayView1<f32>) -> (f32, Array1<f32>) {
    // OPTIMIZATION: Compute sum_sq with iterator for SIMD
    let sum_sq: f32 = x
      .iter()
      .zip(y.iter())
      .map(|(a, b)| {
        let diff = a - b;
        diff * diff
      })
      .sum();
    let dist = sum_sq.sqrt();

    // OPTIMIZATION: Compute gradient with map for better vectorization
    let denom = dist + 1e-6;
    let grad = x
      .iter()
      .zip(y.iter())
      .map(|(a, b)| (a - b) / denom)
      .collect();

    (dist, grad)
  }

  fn disconnection_threshold(&self) -> f32 {
    f32::INFINITY
  }

  /// Provides optimized squared Euclidean distance (avoids sqrt).
  ///
  /// This enables the specialized Euclidean optimization path which is
  /// significantly faster than the generic metric path.
  fn squared_distance(&self, x: ArrayView1<f32>, y: ArrayView1<f32>) -> Option<f32> {
    Some(rdist(&x, &y))
  }
}

/// Squared Euclidean distance (rdist) - used in euclidean optimization for speed
/// This avoids the sqrt operation
/// OPTIMIZATION: Inline always and use iterator for better auto-vectorization
#[inline(always)]
pub fn rdist(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
  // Using iterator allows better SIMD auto-vectorization
  x.iter()
    .zip(y.iter())
    .map(|(a, b)| {
      let diff = a - b;
      diff * diff
    })
    .sum()
}
