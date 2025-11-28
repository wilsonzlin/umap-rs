use crate::metric::Metric;
use crate::metric::MetricType;
use ndarray::Array1;
use ndarray::ArrayView1;

/// Euclidean (L2) distance metric.
///
/// Computes the standard Euclidean distance: sqrt(sum((x_i - y_i)^2))
/// Provides optimized squared distance computation for faster optimization.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
  fn metric_type(&self) -> MetricType {
    MetricType::Euclidean
  }

  /// Compute Euclidean distance and its gradient.
  ///
  /// Returns (distance, gradient) where gradient = (x - y) / (distance + ε)
  /// The epsilon term prevents division by zero when points are identical.
  fn distance(&self, x: ArrayView1<f32>, y: ArrayView1<f32>) -> (f32, Array1<f32>) {
    let mut sum_sq = 0.0;
    for i in 0..x.len() {
      let diff = x[i] - y[i];
      sum_sq += diff * diff;
    }
    let dist = sum_sq.sqrt();

    let mut grad = Array1::zeros(x.len());
    let denom = dist + 1e-6;
    for i in 0..x.len() {
      grad[i] = (x[i] - y[i]) / denom;
    }

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
#[inline]
pub fn rdist(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
  let mut sum_sq = 0.0;
  for i in 0..x.len() {
    let diff = x[i] - y[i];
    sum_sq += diff * diff;
  }
  sum_sq
}
