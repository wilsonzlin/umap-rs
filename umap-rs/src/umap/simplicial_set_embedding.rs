use crate::layout::optimize_layout_euclidean::OptimizeLayoutEuclidean;
use crate::layout::optimize_layout_generic::OptimizeLayoutGeneric;
use crate::metric::Metric;
use crate::umap::make_epochs_per_sample::make_epochs_per_sample;
use ndarray::Array2;
use ndarray::ArrayView2;
use sprs::CsMatView;
use sprs::TriMat;
use typed_builder::TypedBuilder;

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
    let Self {
      graph,
      initial_alpha,
      a,
      b,
      gamma,
      negative_sample_rate,
      n_epochs,
      init,
      output_metric,
    } = self;

    // Check if metric provides fast squared distance (indicates Euclidean-like metric)
    let euclidean_output = output_metric
      .squared_distance(
        ndarray::ArrayView1::from(&[0.0]),
        ndarray::ArrayView1::from(&[0.0]),
      )
      .is_some();

    // Convert to CSR if not already
    let graph = graph.to_csr();
    let n_vertices = graph.shape().1;

    // For smaller datasets we can use more epochs
    let default_epochs = if graph.shape().0 <= 10000 { 500 } else { 200 };

    let n_epochs = n_epochs.unwrap_or(default_epochs);
    let n_epochs_max = n_epochs;

    // Find max value in graph
    // OPTIMIZATION: Use fold for better performance
    let max_val = graph
      .data()
      .iter()
      .copied()
      .fold(f32::NEG_INFINITY, f32::max)
      .max(1.0);

    // Filter weak edges
    let threshold = if n_epochs_max > 10 {
      max_val / n_epochs_max as f32
    } else {
      max_val / default_epochs as f32
    };

    // Build filtered graph using TriMat
    let mut tri = TriMat::new(graph.shape());
    for (&val, (row, col)) in graph.iter() {
      if val >= threshold {
        tri.add_triplet(row, col, val);
      }
    }
    let graph = tri.to_csr::<usize>();

    let mut embedding = init.to_owned();

    // Convert graph data to f32 array view
    let graph_data_vec: Vec<f32> = graph.data().to_vec();
    let graph_data_view = ndarray::Array1::from(graph_data_vec);

    let epochs_per_sample = make_epochs_per_sample(&graph_data_view.view(), n_epochs_max);

    // Extract head and tail indices from graph structure
    let mut head = Vec::new();
    let mut tail = Vec::new();

    for (row_idx, row) in graph.outer_iterator().enumerate() {
      for (&col_idx, _val) in std::iter::zip(row.indices(), row.data()) {
        head.push(row_idx as u32);
        tail.push(col_idx as u32);
      }
    }

    let head_array = ndarray::Array1::from(head);
    let tail_array = ndarray::Array1::from(tail);

    // Normalize embedding to be in range [0, 10]
    // OPTIMIZATION: Use fold for more efficient min/max finding
    for col in 0..embedding.shape()[1] {
      let col_view = embedding.column(col);

      // Find min and max in a single pass
      let (min, max) = col_view
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), val| {
          (min.min(val), max.max(val))
        });

      let range = max - min;
      if range > 0.0 {
        let scale = 10.0 / range;
        for row in 0..embedding.shape()[0] {
          embedding[(row, col)] = (embedding[(row, col)] - min) * scale;
        }
      }
    }

    if euclidean_output {
      let mut embedding_copy = embedding.clone();
      OptimizeLayoutEuclidean::builder()
        .head_embedding(&mut embedding.view_mut())
        .tail_embedding(&mut embedding_copy.view_mut())
        .head(&head_array.view())
        .tail(&tail_array.view())
        .n_epochs(n_epochs_max)
        .n_vertices(n_vertices)
        .epochs_per_sample(&epochs_per_sample.view())
        .a(a)
        .b(b)
        .gamma(gamma)
        .initial_alpha(initial_alpha)
        .negative_sample_rate(negative_sample_rate as f64)
        .parallel(true)
        .move_other(true)
        .build()
        .exec();
    } else {
      let mut embedding_copy = embedding.clone();
      OptimizeLayoutGeneric::builder()
        .head_embedding(&mut embedding.view_mut())
        .tail_embedding(&mut embedding_copy.view_mut())
        .head(&head_array.view())
        .tail(&tail_array.view())
        .n_epochs(n_epochs_max)
        .n_vertices(n_vertices)
        .epochs_per_sample(&epochs_per_sample.view())
        .a(a)
        .b(b)
        .gamma(gamma)
        .initial_alpha(initial_alpha)
        .negative_sample_rate(negative_sample_rate as f64)
        .output_metric(output_metric)
        .move_other(true)
        .build()
        .exec();
    }

    embedding
  }
}
