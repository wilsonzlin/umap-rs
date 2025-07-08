use ndarray::ArrayView2;
use sprs::CsMatView;
use typed_builder::TypedBuilder;

use crate::metric::Metric;

#[derive(Debug, TypedBuilder)]
pub struct SimplicialSetEmbedding<'a> {
  graph: &'a CsMatView<'a, f32>,
  initial_alpha: f32,
  a: f32,
  b: f32,
  gamma: f32,
  negative_sample_rate: usize,
  n_epochs: usize,
  init: &'a ArrayView2<'a, f32>,
  output_metric: &'a dyn Metric,
  #[builder(default=true)]
  euclidean_output: bool,
}

impl<'a> SimplicialSetEmbedding<'a> {
  pub fn exec(self) {
    let Self { graph, initial_alpha, a, b, gamma, negative_sample_rate, n_epochs, init, output_metric, euclidean_output } = self;

    graph = graph.tocoo();
    graph.sum_duplicates();
    n_vertices = graph.shape[1];

    // For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000 {
        default_epochs = 500
    } else {
        default_epochs = 200
    }

    if n_epochs is None:
        n_epochs = default_epochs

    // If n_epoch is a list, get the maximum epoch to reach
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

    if n_epochs_max > 10 {
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    } else {
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0
    }

    graph.eliminate_zeros();

    init_data = np.array(init);
    embedding = init_data;

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max);

    head = graph.row;
    tail = graph.col;
    weight = graph.data;

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64);

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    if euclidean_output {
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
    } else {
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
            move_other=True,
        )
    }

    return embedding
  }
}
