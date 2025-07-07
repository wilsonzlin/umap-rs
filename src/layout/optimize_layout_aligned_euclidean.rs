def _optimize_layout_aligned_euclidean_single_epoch(
  head_embeddings,
  tail_embeddings,
  heads,
  tails,
  epochs_per_sample,
  a,
  b,
  regularisation_weights,
  relations,
  rng_state,
  gamma,
  lambda_,
  dim,
  move_other,
  alpha,
  epochs_per_negative_sample,
  epoch_of_next_negative_sample,
  epoch_of_next_sample,
  n,
):
  n_embeddings = len(heads)
  window_size = (relations.shape[1] - 1) // 2

  max_n_edges = 0
  for e_p_s in epochs_per_sample:
      if e_p_s.shape[0] >= max_n_edges:
          max_n_edges = e_p_s.shape[0]

  embedding_order = np.arange(n_embeddings).astype(np.int32)
  np.random.seed(abs(rng_state[0]))
  np.random.shuffle(embedding_order)

  for i in range(max_n_edges):
      for m in embedding_order:
          if i < epoch_of_next_sample[m].shape[0] and epoch_of_next_sample[m][i] <= n:
              j = heads[m][i]
              k = tails[m][i]

              current = head_embeddings[m][j]
              other = tail_embeddings[m][k]

              dist_squared = rdist(current, other)

              if dist_squared > 0.0:
                  grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                  grad_coeff /= a * pow(dist_squared, b) + 1.0
              else:
                  grad_coeff = 0.0

              for d in range(dim):
                  grad_d = clip(grad_coeff * (current[d] - other[d]))

                  for offset in range(-window_size, window_size):
                      neighbor_m = m + offset
                      if n_embeddings > neighbor_m >= 0 != offset:
                          identified_index = relations[m, offset + window_size, j]
                          if identified_index >= 0:
                              grad_d -= clip(
                                  (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                  * regularisation_weights[m, offset + window_size, j]
                                  * (
                                      current[d]
                                      - head_embeddings[neighbor_m][
                                          identified_index, d
                                      ]
                                  )
                              )

                  current[d] += clip(grad_d) * alpha
                  if move_other:
                      other_grad_d = clip(grad_coeff * (other[d] - current[d]))

                      for offset in range(-window_size, window_size):
                          neighbor_m = m + offset
                          if n_embeddings > neighbor_m >= 0 != offset:
                              identified_index = relations[m, offset + window_size, k]
                              if identified_index >= 0:
                                  other_grad_d -= clip(
                                      (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                      * regularisation_weights[
                                          m, offset + window_size, k
                                      ]
                                      * (
                                          other[d]
                                          - head_embeddings[neighbor_m][
                                              identified_index, d
                                          ]
                                      )
                                  )

                      other[d] += clip(other_grad_d) * alpha

              epoch_of_next_sample[m][i] += epochs_per_sample[m][i]

              if epochs_per_negative_sample[m][i] > 0:
                  n_neg_samples = int(
                      (n - epoch_of_next_negative_sample[m][i])
                      / epochs_per_negative_sample[m][i]
                  )
              else:
                  n_neg_samples = 0

              for p in range(n_neg_samples):
                  k = tau_rand_int(rng_state) % tail_embeddings[m].shape[0]

                  other = tail_embeddings[m][k]

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
                          grad_d = 0.0

                      for offset in range(-window_size, window_size):
                          neighbor_m = m + offset
                          if n_embeddings > neighbor_m >= 0 != offset:
                              identified_index = relations[m, offset + window_size, j]
                              if identified_index >= 0:
                                  grad_d -= clip(
                                      (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                      * regularisation_weights[
                                          m, offset + window_size, j
                                      ]
                                      * (
                                          current[d]
                                          - head_embeddings[neighbor_m][
                                              identified_index, d
                                          ]
                                      )
                                  )

                      current[d] += clip(grad_d) * alpha

              epoch_of_next_negative_sample[m][i] += (
                  n_neg_samples * epochs_per_negative_sample[m][i]
              )

def optimize_layout_aligned_euclidean(
  head_embeddings,
  tail_embeddings,
  heads,
  tails,
  n_epochs,
  epochs_per_sample,
  regularisation_weights,
  relations,
  rng_state,
  a=1.576943460405378,
  b=0.8950608781227859,
  gamma=1.0,
  lambda_=5e-3,
  initial_alpha=1.0,
  negative_sample_rate=5.0,
  parallel=True,
  verbose=False,
  tqdm_kwds=None,
  move_other=False,
):
  dim = head_embeddings[0].shape[1]
  alpha = initial_alpha

  epochs_per_negative_sample = numba.typed.List.empty_list(numba.types.float32[::1])
  epoch_of_next_negative_sample = numba.typed.List.empty_list(
      numba.types.float32[::1]
  )
  epoch_of_next_sample = numba.typed.List.empty_list(numba.types.float32[::1])

  for m in range(len(heads)):
      epochs_per_negative_sample.append(
          epochs_per_sample[m].astype(np.float32) / negative_sample_rate
      )
      epoch_of_next_negative_sample.append(
          epochs_per_negative_sample[m].astype(np.float32)
      )
      epoch_of_next_sample.append(epochs_per_sample[m].astype(np.float32))

  optimize_fn = numba.njit(
      _optimize_layout_aligned_euclidean_single_epoch,
      fastmath=True,
      parallel=parallel,
  )

  if tqdm_kwds is None:
      tqdm_kwds = {}

  if "disable" not in tqdm_kwds:
      tqdm_kwds["disable"] = not verbose

  for n in tqdm(range(n_epochs), **tqdm_kwds):
      optimize_fn(
          head_embeddings,
          tail_embeddings,
          heads,
          tails,
          epochs_per_sample,
          a,
          b,
          regularisation_weights,
          relations,
          rng_state,
          gamma,
          lambda_,
          dim,
          move_other,
          alpha,
          epochs_per_negative_sample,
          epoch_of_next_negative_sample,
          epoch_of_next_sample,
          n,
      )

      alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

  return head_embeddings
