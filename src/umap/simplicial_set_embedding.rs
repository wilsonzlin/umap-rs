def simplicial_set_embedding(
  graph,
  initial_alpha,
  a,
  b,
  gamma,
  negative_sample_rate,
  n_epochs,
  init,
  random_state,
  output_metric=dist.named_distances_with_gradients["euclidean"],
  output_metric_kwds={},
  euclidean_output=True,
):
  graph = graph.tocoo()
  graph.sum_duplicates()
  n_vertices = graph.shape[1]

  # For smaller datasets we can use more epochs
  if graph.shape[0] <= 10000:
      default_epochs = 500
  else:
      default_epochs = 200

  if n_epochs is None:
      n_epochs = default_epochs

  # If n_epoch is a list, get the maximum epoch to reach
  n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

  if n_epochs_max > 10:
      graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
  else:
      graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

  graph.eliminate_zeros()

  init_data = np.array(init)
  if len(init_data.shape) == 2:
      if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
          tree = KDTree(init_data)
          dist, ind = tree.query(init_data, k=2)
          nndist = np.mean(dist[:, 1])
          embedding = init_data + random_state.normal(
              scale=0.001 * nndist, size=init_data.shape
          ).astype(np.float32)
      else:
          embedding = init_data

  epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max)

  head = graph.row
  tail = graph.col
  weight = graph.data

  rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

  aux_data = {}

  embedding = (
      10.0
      * (embedding - np.min(embedding, 0))
      / (np.max(embedding, 0) - np.min(embedding, 0))
  ).astype(np.float32, order="C")

  if euclidean_output:
      embedding = optimize_layout_euclidean(
          embedding,
          embedding,
          head,
          tail,
          n_epochs,
          n_vertices,
          epochs_per_sample,
          a,
          b,
          rng_state,
          gamma,
          initial_alpha,
          negative_sample_rate,
          parallel=parallel,
          move_other=True,
      )
  else:
      embedding = optimize_layout_generic(
          embedding,
          embedding,
          head,
          tail,
          n_epochs,
          n_vertices,
          epochs_per_sample,
          a,
          b,
          rng_state,
          gamma,
          initial_alpha,
          negative_sample_rate,
          output_metric,
          tuple(output_metric_kwds.values()),
          move_other=True,
      )

  if isinstance(embedding, list):
      aux_data["embedding_list"] = embedding
      embedding = embedding[-1].copy()

  return embedding, aux_data
