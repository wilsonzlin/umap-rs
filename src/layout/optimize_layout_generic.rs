
def _optimize_layout_generic_single_epoch(
  epochs_per_sample,
  epoch_of_next_sample,
  head,
  tail,
  head_embedding,
  tail_embedding,
  output_metric,
  output_metric_kwds,
  dim,
  alpha,
  move_other,
  n,
  epoch_of_next_negative_sample,
  epochs_per_negative_sample,
  rng_state_per_sample,
  n_vertices,
  a,
  b,
  gamma,
):
  for i in range(epochs_per_sample.shape[0]):
      if epoch_of_next_sample[i] <= n:
          j = head[i]
          k = tail[i]

          current = head_embedding[j]
          other = tail_embedding[k]

          dist_output, grad_dist_output = output_metric(
              current, other, *output_metric_kwds
          )
          _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)

          if dist_output > 0.0:
              w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
          else:
              w_l = 1.0
          grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

          for d in range(dim):
              grad_d = clip(grad_coeff * grad_dist_output[d])

              current[d] += grad_d * alpha
              if move_other:
                  grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                  other[d] += grad_d * alpha

          epoch_of_next_sample[i] += epochs_per_sample[i]

          n_neg_samples = int(
              (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
          )

          for p in range(n_neg_samples):
              k = tau_rand_int(rng_state_per_sample[j]) % n_vertices

              other = tail_embedding[k]

              dist_output, grad_dist_output = output_metric(
                  current, other, *output_metric_kwds
              )

              if dist_output > 0.0:
                  w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
              elif j == k:
                  continue
              else:
                  w_l = 1.0

              grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

              for d in range(dim):
                  grad_d = clip(grad_coeff * grad_dist_output[d])
                  current[d] += grad_d * alpha

          epoch_of_next_negative_sample[i] += (
              n_neg_samples * epochs_per_negative_sample[i]
          )
  return epoch_of_next_sample, epoch_of_next_negative_sample

  
def optimize_layout_generic(
  head_embedding,
  tail_embedding,
  head,
  tail,
  n_epochs,
  n_vertices,
  epochs_per_sample,
  a,
  b,
  rng_state,
  gamma=1.0,
  initial_alpha=1.0,
  negative_sample_rate=5.0,
  output_metric=dist.euclidean,
  output_metric_kwds=(),
  move_other=False,
):
  """Improve an embedding using stochastic gradient descent to minimize the
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

  rng_state: array of int64, shape (3,)
      The internal state of the rng

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
  """

  dim = head_embedding.shape[1]
  alpha = initial_alpha

  epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
  epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
  epoch_of_next_sample = epochs_per_sample.copy()

  optimize_fn = numba.njit(
      _optimize_layout_generic_single_epoch,
      fastmath=True,
  )

  rng_state_per_sample = np.full(
      (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
  ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

  for n in tqdm(range(n_epochs)):
      optimize_fn(
          epochs_per_sample,
          epoch_of_next_sample,
          head,
          tail,
          head_embedding,
          tail_embedding,
          output_metric,
          output_metric_kwds,
          dim,
          alpha,
          move_other,
          n,
          epoch_of_next_negative_sample,
          epochs_per_negative_sample,
          rng_state_per_sample,
          n_vertices,
          a,
          b,
          gamma,
      )
      alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

  return head_embedding
