use ndarray::ArrayView2;
use sprs::CsMatView;
use typed_builder::TypedBuilder;

use crate::{metric::Metric, umap::make_epochs_per_sample::make_epochs_per_sample};

#[derive(Debug, TypedBuilder)]
pub struct SimplicialSetEmbedding<'g, 'i, 'o> {
  graph: CsMatView<'g, f32>,
  initial_alpha: f32,
  a: f32,
  b: f32,
  gamma: f32,
  negative_sample_rate: usize,
  n_epochs: Option<usize>,
  init: ArrayView2<'i, f32>,
  output_metric: &'o dyn Metric,
}

impl<'g, 'i, 'o> SimplicialSetEmbedding<'g, 'i, 'o> {
  pub fn exec(self) {
    let Self { graph, initial_alpha, a, b, gamma, negative_sample_rate, n_epochs, init, output_metric } = self;
    let euclidean_output = output_metric.is_euclidean();

    let graph = graph.tocoo();
    graph.sum_duplicates();
    let n_vertices = graph.shape().1;

    // For smaller datasets we can use more epochs
    let default_epochs = if graph.shape().0 <= 10000 {
        500
    } else {
        200
    };

    let n_epochs = n_epochs.unwrap_or(default_epochs);

    let n_epochs_max = n_epochs;

    if n_epochs_max > 10 {
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    } else {
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0
    }

    graph.eliminate_zeros();

    let mut embedding = init.to_owned();

    let epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max);

    let head = graph.row;
    let tail = graph.col;
    let weight = graph.data;

    embedding =
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0));

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

    embedding
  }
}
