
def _optimize_layout_euclidean_single_epoch(
  head_embedding,
  tail_embedding,
  head,
  tail,
  n_vertices,
  epochs_per_sample,
  a,
  b,
  rng_state_per_sample,
  gamma,
  dim,
  move_other,
  alpha,
  epochs_per_negative_sample,
  epoch_of_next_negative_sample,
  epoch_of_next_sample,
  n,
):
  for i in numba.prange(epochs_per_sample.shape[0]):
      if epoch_of_next_sample[i] <= n:
          j = head[i]
          k = tail[i]

          current = head_embedding[j]
          other = tail_embedding[k]

          dist_squared = rdist(current, other)

          if dist_squared > 0.0:
              grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
              grad_coeff /= a * pow(dist_squared, b) + 1.0
          else:
              grad_coeff = 0.0

          for d in range(dim):
              grad_d = clip(grad_coeff * (current[d] - other[d]))

              current[d] += grad_d * alpha
              if move_other:
                  other[d] += -grad_d * alpha

          epoch_of_next_sample[i] += epochs_per_sample[i]

          n_neg_samples = int(
              (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
          )

          for p in range(n_neg_samples):
              k = tau_rand_int(rng_state_per_sample[j]) % n_vertices

              other = tail_embedding[k]

              dist_squared = rdist(current, other)

              if dist_squared > 0.0:
                  grad_coeff = 2.0 * gamma * b
                  grad_coeff /= (0.001 + dist_squared) * (
                      a * pow(dist_squared, b) + 1
                  )
              elif j == k:
                  continue
              else:
                  grad_coeff = 0.0

              for d in range(dim):
                  if grad_coeff > 0.0:
                      grad_d = clip(grad_coeff * (current[d] - other[d]))
                  else:
                      grad_d = 0
                  current[d] += grad_d * alpha

          epoch_of_next_negative_sample[i] += (
              n_neg_samples * epochs_per_negative_sample[i]
          )


_nb_optimize_layout_euclidean_single_epoch = numba.njit(
  _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=False
)

_nb_optimize_layout_euclidean_single_epoch_parallel = numba.njit(
  _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=True
)


def _get_optimize_layout_euclidean_single_epoch_fn(parallel: bool = False):
  if parallel:
      return _nb_optimize_layout_euclidean_single_epoch_parallel
  else:
      return _nb_optimize_layout_euclidean_single_epoch


def optimize_layout_euclidean(
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
  parallel=False,
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
  n_epochs: int, or list of int
      The number of training epochs to use in optimization, or a list of
      epochs at which to save the embedding. In case of a list, the optimization
      will use the maximum number of epochs in the list, and will return a list
      of embedding in the order of increasing epoch, regardless of the order in
      the epoch list.
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
  parallel: bool (optional, default False)
      Whether to run the computation using numba parallel.
      Running in parallel is non-deterministic, and is not used
      if a random seed has been set, to ensure reproducibility.
  verbose: bool (optional, default False)
      Whether to report information on the current progress of the algorithm.
  densmap: bool (optional, default False)
      Whether to use the density-augmented densMAP objective
  densmap_kwds: dict (optional, default None)
      Auxiliary data for densMAP
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

  # Fix for calling UMAP many times for small datasets, otherwise we spend here
  # a lot of time in compilation step (first call to numba function)
  optimize_fn = _get_optimize_layout_euclidean_single_epoch_fn(parallel)

  epochs_list = None
  embedding_list = []
  if isinstance(n_epochs, list):
      epochs_list = n_epochs
      n_epochs = max(epochs_list)

  rng_state_per_sample = np.full(
      (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
  ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

  for n in tqdm(range(n_epochs)):
      optimize_fn(
          head_embedding,
          tail_embedding,
          head,
          tail,
          n_vertices,
          epochs_per_sample,
          a,
          b,
          rng_state_per_sample,
          gamma,
          dim,
          move_other,
          alpha,
          epochs_per_negative_sample,
          epoch_of_next_negative_sample,
          epoch_of_next_sample,
          n,
      )

      alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

      if epochs_list is not None and n in epochs_list:
          embedding_list.append(head_embedding.copy())

  # Add the last embedding to the list as well
  if epochs_list is not None:
      embedding_list.append(head_embedding.copy())

  return head_embedding if epochs_list is None else embedding_list
