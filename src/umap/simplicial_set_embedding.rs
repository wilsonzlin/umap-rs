use ndarray::{Array2, ArrayView2, Axis};
use sprs::CsMatView;
use typed_builder::TypedBuilder;

use crate::metric::Metric;
use crate::umap::make_epochs_per_sample::make_epochs_per_sample;

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
  pub fn exec(self) -> Array2<f32> {
    let Self { graph, initial_alpha, a, b, gamma, negative_sample_rate, n_epochs, init, output_metric } = self;
    let _euclidean_output = output_metric.is_euclidean();

    let graph_csr = graph.to_csr();
    let n_vertices = graph.cols();

    // For smaller datasets we can use more epochs
    let default_epochs = if graph.rows() <= 10000 {
        500
    } else {
        200
    };

    let n_epochs = n_epochs.unwrap_or(default_epochs);

    let n_epochs_max = n_epochs;

    // Find max value for filtering
    let graph_max = graph_csr.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let threshold = if n_epochs_max > 10 {
        graph_max / (n_epochs_max as f32)
    } else {
        graph_max / (default_epochs as f32)
    };

    // Build filtered edge list
    let mut filtered_data = Vec::new();

    for (row_idx, row) in graph_csr.outer_iterator().enumerate() {
        for (&col_idx, &value) in row.indices().iter().zip(row.data().iter()) {
            if value > threshold {
                filtered_data.push(value as f64);
            }
        }
    }

    let mut embedding = init.to_owned();

    let epochs_per_sample = make_epochs_per_sample(
        &ndarray::ArrayView1::from(&filtered_data),
        n_epochs_max
    );

    // Normalize embedding to [0, 10] range per dimension
    let min_vals = embedding.fold_axis(Axis(0), f32::INFINITY, |&a, &b| a.min(b));
    let max_vals = embedding.fold_axis(Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b));

    for i in 0..embedding.shape()[0] {
        for j in 0..embedding.shape()[1] {
            let range = max_vals[j] - min_vals[j];
            if range > 0.0 {
                embedding[[i, j]] = 10.0 * (embedding[[i, j]] - min_vals[j]) / range;
            }
        }
    }

    // TODO: Call optimize_layout_euclidean or optimize_layout_generic here
    // For now, just return the initialized embedding
    // This will be completed when we port the euclidean optimization

    embedding
  }
}
