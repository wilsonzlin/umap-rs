use ndarray::{Array1, ArrayView1};

use crate::metric::Metric;

/// Euclidean distance metric with gradient computation
#[derive(Debug, Clone)]
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
    /// Compute Euclidean distance and its gradient
    /// Returns (distance, gradient) where gradient = (x - y) / (distance + epsilon)
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
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

    fn is_euclidean(&self) -> bool {
        true
    }

    fn default_disconnection_distance(&self) -> f32 {
        f32::INFINITY
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
