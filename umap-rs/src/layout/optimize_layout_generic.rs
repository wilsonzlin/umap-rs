use crate::metric::Metric;
use crate::utils::clip::clip;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut2;
use rand::Rng;

/// Execute a single epoch of generic (metric-agnostic) UMAP optimization.
///
/// Performs one epoch of stochastic gradient descent to optimize the embedding,
/// using an arbitrary distance metric. This is slower than the Euclidean
/// specialization but supports any metric implementing the Metric trait.
///
/// The epoch state arrays are updated in-place, allowing for stateful optimization
/// with checkpointing support.
///
/// # Arguments
///
/// * `head_embedding` - The embedding being optimized (mutated in-place)
/// * `tail_embedding` - Reference embedding (usually same as head for standard UMAP)
/// * `head` - Head vertex indices for each edge
/// * `tail` - Tail vertex indices for each edge
/// * `n_vertices` - Total number of vertices in the dataset
/// * `epochs_per_sample` - Sampling frequency for each edge
/// * `a`, `b` - Curve parameters for distance-to-probability transformation
/// * `gamma` - Repulsion strength for negative samples
/// * `alpha` - Current learning rate for this epoch
/// * `epochs_per_negative_sample` - Negative sampling frequency per edge
/// * `epoch_of_next_sample` - Next epoch to sample each positive edge (mutated)
/// * `epoch_of_next_negative_sample` - Next epoch for negative sampling (mutated)
/// * `current_epoch` - Current epoch number
/// * `move_other` - Whether to update tail_embedding as well
/// * `output_metric` - The distance metric to use for optimization
#[allow(clippy::too_many_arguments)]
pub fn optimize_layout_generic_single_epoch_stateful(
  head_embedding: &mut ArrayViewMut2<f32>,
  tail_embedding: &mut ArrayViewMut2<f32>,
  head: &ArrayView1<u32>,
  tail: &ArrayView1<u32>,
  n_vertices: usize,
  epochs_per_sample: &ArrayView1<f64>,
  a: f32,
  b: f32,
  gamma: f32,
  alpha: f32,
  epochs_per_negative_sample: &mut Array1<f64>,
  epoch_of_next_sample: &mut Array1<f64>,
  epoch_of_next_negative_sample: &mut Array1<f64>,
  current_epoch: usize,
  move_other: bool,
  output_metric: &dyn Metric,
) {
  let dim = head_embedding.shape()[1];
  let mut rng = rand::rng();

  for i in 0..epochs_per_sample.len() {
    if epoch_of_next_sample[i] <= current_epoch as f64 {
      let j = head[i] as usize;
      let k = tail[i] as usize;

      // Positive sample: attract neighbors
      let current = head_embedding.row(j);
      let other = tail_embedding.row(k);

      let (dist_output, grad_dist_output) = output_metric.distance(current, other);
      let (_, rev_grad_dist_output) = output_metric.distance(other, current);

      let mut current = head_embedding.row_mut(j);
      let mut other = tail_embedding.row_mut(k);

      let w_l = if dist_output > 0.0 {
        f32::powi(1.0 + a * f32::powf(dist_output, 2.0 * b), -1)
      } else {
        1.0
      };
      let grad_coeff = 2.0 * b * (w_l - 1.0) / (dist_output + 1e-6);

      for d in 0..dim {
        let mut grad_d = clip(grad_coeff * grad_dist_output[d]);
        current[d] += grad_d * alpha;

        if move_other {
          grad_d = clip(grad_coeff * rev_grad_dist_output[d]);
          other[d] += grad_d * alpha;
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      // Negative samples: repel non-neighbors
      let n_neg_samples = ((current_epoch as f64 - epoch_of_next_negative_sample[i])
        / epochs_per_negative_sample[i]) as usize;

      for _p in 0..n_neg_samples {
        let k = rng.random_range(0..n_vertices);
        if j == k {
          continue;
        }

        let current = head_embedding.row(j);
        let other = tail_embedding.row(k);

        let (dist_output, grad_dist_output) = output_metric.distance(current, other);

        let mut current = head_embedding.row_mut(j);

        let w_l = if dist_output > 0.0 {
          f32::powi(1.0 + a * f32::powf(dist_output, 2.0 * b), -1)
        } else {
          1.0
        };

        let grad_coeff = gamma * 2.0 * b * w_l / (dist_output + 1e-6);

        for d in 0..dim {
          let grad_d = clip(grad_coeff * grad_dist_output[d]);
          current[d] += grad_d * alpha;
        }
      }

      epoch_of_next_negative_sample[i] += n_neg_samples as f64 * epochs_per_negative_sample[i];
    }
  }
}
