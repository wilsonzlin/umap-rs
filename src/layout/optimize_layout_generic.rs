use ndarray::{ArrayView1, ArrayViewMut1, ArrayViewMut2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use typed_builder::TypedBuilder;

use crate::metric::Metric;
use crate::utils::clip::clip;

#[derive(TypedBuilder)]
struct OptimizeLayoutGenericSingleEpoch<'a> {
  epochs_per_sample: &'a ArrayView1<'a, f64>,
  epoch_of_next_sample: &'a mut ArrayViewMut1<'a, f64>,
  head: &'a ArrayView1<'a, usize>,
  tail: &'a ArrayView1<'a, usize>,
  head_embedding: &'a mut ArrayViewMut2<'a, f32>,
  tail_embedding: &'a mut ArrayViewMut2<'a, f32>,
  output_metric: &'a dyn Metric,
  dim: usize,
  alpha: f32,
  move_other: bool,
  n: usize,
  epoch_of_next_negative_sample: &'a mut ArrayViewMut1<'a, f64>,
  epochs_per_negative_sample: &'a ArrayView1<'a, f64>,
  n_vertices: usize,
  a: f32,
  b: f32,
  gamma: f32,
  rng: &'a mut StdRng,
}

impl<'a> OptimizeLayoutGenericSingleEpoch<'a> {
  pub fn exec(self) {
    let Self {
      epochs_per_sample,
      epoch_of_next_sample,
      head,
      tail,
      head_embedding,
      tail_embedding,
      output_metric,
      dim,
      alpha,
      move_other,
      n,
      epoch_of_next_negative_sample,
      epochs_per_negative_sample,
      n_vertices,
      a,
      b,
      gamma,
      rng,
    } = self;

    for i in 0..epochs_per_sample.len() {
        if epoch_of_next_sample[i] <= n as f64 {
            let j = head[i];
            let k = tail[i];

            let current = head_embedding.row(j);
            let other = tail_embedding.row(k);

            let (dist_output, grad_dist_output) = output_metric.distance(&current, &other);
            let (_, rev_grad_dist_output) = output_metric.distance(&other, &current);

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

            let n_neg_samples = (
                (n as f64 - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            ) as usize;

            for _p in 0..n_neg_samples {
                let k = rng.random_range(0..n_vertices);

                let current = head_embedding.row(j);
                let other = tail_embedding.row(k);

                let (dist_output, grad_dist_output) = output_metric.distance(
                    &current, &other
                );

                let mut current = head_embedding.row_mut(j);

                let w_l = if dist_output > 0.0 {
                    f32::powi(1.0 + a * f32::powf(dist_output, 2.0 * b), -1)
                } else if j == k {
                    continue
                } else {
                    1.0
                };

                let grad_coeff = gamma * 2.0 * b * w_l / (dist_output + 1e-6);

                for d in 0..dim {
                    let grad_d = clip(grad_coeff * grad_dist_output[d]);
                    current[d] += grad_d * alpha;
                }
            }

            epoch_of_next_negative_sample[i] +=
                n_neg_samples as f64 * epochs_per_negative_sample[i];
        };
    }
  }
}

/*
  Improve an embedding using stochastic gradient descent to minimize the
  fuzzy set cross entropy between the 1-skeletons of the high dimensional
  and low dimensional fuzzy simplicial sets. In practice this is done by
  sampling edges based on their membership strength (with the (1-p) terms
  coming from negative sampling similar to word2vec).

  Parameters
  ----------
  head_embedding: array of shape (n_samples, n_components)
      The initial embedding to be improved by SGD.

  tail_embedding: array of shape (source_samples, n_components)
      The reference embedding of embedded points. If not embedding new
      previously unseen points with respect to an existing embedding this
      is simply the head_embedding (again); otherwise it provides the
      existing embedding to embed with respect to.

  head: array of shape (n_1_simplices)
      The indices of the heads of 1-simplices with non-zero membership.

  tail: array of shape (n_1_simplices)
      The indices of the tails of 1-simplices with non-zero membership.

  n_epochs: int
      The number of training epochs to use in optimization.

  n_vertices: int
      The number of vertices (0-simplices) in the dataset.

  epochs_per_sample: array of shape (n_1_simplices)
      A float value of the number of epochs per 1-simplex. 1-simplices with
      weaker membership strength will have more epochs between being sampled.

  a: float
      Parameter of differentiable approximation of right adjoint functor

  b: float
      Parameter of differentiable approximation of right adjoint functor

  gamma: float (optional, default 1.0)
      Weight to apply to negative samples.

  initial_alpha: float (optional, default 1.0)
      Initial learning rate for the SGD.

  negative_sample_rate: int (optional, default 5)
      Number of negative samples to use per positive sample.

  verbose: bool (optional, default False)
      Whether to report information on the current progress of the algorithm.

  tqdm_kwds: dict (optional, default None)
      Keyword arguments for tqdm progress bar.

  move_other: bool (optional, default False)
      Whether to adjust tail_embedding alongside head_embedding

  Returns
  -------
  embedding: array of shape (n_samples, n_components)
      The optimized embedding.
*/
#[derive(TypedBuilder)]
pub struct OptimizeLayoutGeneric<'a> {
  head_embedding: &'a mut ArrayViewMut2<'a, f32>,
  tail_embedding: &'a mut ArrayViewMut2<'a, f32>,
  head: &'a ArrayView1<'a, usize>,
  tail: &'a ArrayView1<'a, usize>,
  n_epochs: usize,
  n_vertices: usize,
  epochs_per_sample: &'a ArrayView1<'a, f64>,
  a: f32,
  b: f32,
  #[builder(default = 1.0)]
  gamma: f32,
  #[builder(default = 1.0)]
  initial_alpha: f32,
  #[builder(default = 5.0)]
  negative_sample_rate: f64,
  output_metric: &'a dyn Metric,
  #[builder(default = 42)]
  random_seed: u64,
  #[builder(default = false)]
  move_other: bool,
}

impl<'a> OptimizeLayoutGeneric<'a> {
  pub fn exec(self) -> &'a mut ArrayViewMut2<'a, f32> {
    let Self {
      head_embedding,
      tail_embedding,
      head,
      tail,
      n_epochs,
      n_vertices,
      epochs_per_sample,
      a,
      b,
      gamma,
      initial_alpha,
      negative_sample_rate,
      output_metric,
      random_seed,
      move_other,
    } = self;

    let dim = head_embedding.shape()[1];
    let mut alpha = initial_alpha;

    let epochs_per_negative_sample = epochs_per_sample / negative_sample_rate;
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.to_owned();
    let mut epoch_of_next_sample = epochs_per_sample.to_owned();

    let mut rng = StdRng::seed_from_u64(random_seed);

    for n in 0..n_epochs {
      OptimizeLayoutGenericSingleEpoch::builder()
        .epochs_per_sample(epochs_per_sample)
        .epoch_of_next_sample(&mut epoch_of_next_sample.view_mut())
        .head(head)
        .tail(tail)
        .head_embedding(head_embedding)
        .tail_embedding(tail_embedding)
        .output_metric(output_metric)
        .dim(dim)
        .alpha(alpha)
        .move_other(move_other)
        .n(n)
        .epoch_of_next_negative_sample(&mut epoch_of_next_negative_sample.view_mut())
        .epochs_per_negative_sample(&epochs_per_negative_sample.view())
        .n_vertices(n_vertices)
        .a(a)
        .b(b)
        .gamma(gamma)
        .rng(&mut rng)
        .build()
        .exec();
      alpha = initial_alpha * (1.0 - (n as f32 / n_epochs as f32));
    }

    head_embedding
  }
}
