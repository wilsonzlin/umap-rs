use crate::config::UmapConfig;
use crate::embedding::FittedUmap;
use crate::layout::optimize_layout_euclidean::optimize_layout_euclidean_single_epoch_stateful;
use crate::layout::optimize_layout_generic::optimize_layout_generic_single_epoch_stateful;
use crate::manifold::LearnedManifold;
use crate::metric::Metric;
use crate::metric::MetricType;
use crate::umap::make_epochs_per_sample::make_epochs_per_sample;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use serde::Deserialize;
use serde::Serialize;
use sprs::TriMat;

/// Active optimization state for UMAP embedding.
///
/// This contains all the state needed to run and resume stochastic gradient
/// descent optimization. It's large and mutable, meant to be used during
/// training and then converted to a lightweight `FittedUmap` when done.
///
/// The optimizer can be serialized mid-training to enable fault-tolerant
/// training with checkpoints.
#[derive(Debug, Serialize, Deserialize)]
pub struct Optimizer {
  // Reference to learned manifold
  manifold: LearnedManifold,

  // Preprocessed graph structures for optimization
  head: Array1<u32>,
  tail: Array1<u32>,
  epochs_per_sample: Array1<f64>,

  // Current embedding state
  embedding: Array2<f32>,

  // SGD scheduling state
  epoch_of_next_sample: Array1<f64>,
  epoch_of_next_negative_sample: Array1<f64>,
  epochs_per_negative_sample: Array1<f64>,

  // Progress tracking
  current_epoch: usize,
  total_epochs: usize,

  // Optimization parameters
  gamma: f32,
  initial_alpha: f32,
  negative_sample_rate: f64,

  // Metric type (determined once at creation)
  metric_type: MetricType,
}

impl Optimizer {
  /// Create a new optimizer from a learned manifold.
  ///
  /// This performs preprocessing:
  /// - Filters weak edges from the graph
  /// - Extracts head/tail edge lists
  /// - Computes epoch sampling schedules
  /// - Normalizes the initial embedding to [0, 10]
  ///
  /// # Arguments
  ///
  /// * `manifold` - The learned manifold structure
  /// * `init` - Initial embedding (will be normalized)
  /// * `total_epochs` - Total number of epochs to run
  /// * `opt_params` - Optimization parameters (learning rate, negative sampling, etc.)
  /// * `metric_type` - Type of distance metric being used
  pub fn new(
    manifold: LearnedManifold,
    init: Array2<f32>,
    total_epochs: usize,
    opt_params: &UmapConfig,
    metric_type: MetricType,
  ) -> Self {
    let gamma = opt_params.optimization.repulsion_strength;
    let initial_alpha = opt_params.optimization.learning_rate;
    let negative_sample_rate = opt_params.optimization.negative_sample_rate;

    let graph = &manifold.graph;
    let n_samples = graph.shape().0;

    // Determine epoch threshold for filtering weak edges
    let max_val = graph
      .data()
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap_or(1.0);

    let default_epochs = if n_samples <= 10000 { 500 } else { 200 };
    let threshold_epochs = if total_epochs > 10 {
      total_epochs
    } else {
      default_epochs
    };
    let threshold = max_val / threshold_epochs as f32;

    // Filter weak edges using TriMat
    let mut tri = TriMat::new(graph.shape());
    for (&val, (row, col)) in graph.iter() {
      if val >= threshold {
        tri.add_triplet(row, col, val);
      }
    }
    let filtered_graph = tri.to_csr::<usize>();

    // Compute epochs per sample from edge weights
    let graph_data_vec: Vec<f32> = filtered_graph.data().to_vec();
    let graph_data_array = Array1::from(graph_data_vec);
    let epochs_per_sample = make_epochs_per_sample(&graph_data_array.view(), total_epochs);

    // Extract head and tail indices from graph structure
    let mut head = Vec::new();
    let mut tail = Vec::new();

    for (row_idx, row) in filtered_graph.outer_iterator().enumerate() {
      for (&col_idx, _val) in std::iter::zip(row.indices(), row.data()) {
        head.push(row_idx as u32);
        tail.push(col_idx as u32);
      }
    }

    let head = Array1::from(head);
    let tail = Array1::from(tail);

    // Normalize embedding to [0, 10] range
    let mut embedding = init;
    for col_idx in 0..embedding.shape()[1] {
      let col_view = embedding.column(col_idx);
      let min = col_view
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
      let max = col_view
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);

      let range = max - min;
      if range > 0.0 {
        for row_idx in 0..embedding.shape()[0] {
          embedding[(row_idx, col_idx)] = 10.0 * (embedding[(row_idx, col_idx)] - min) / range;
        }
      }
    }

    // Initialize epoch scheduling
    let epochs_per_negative_sample: Array1<f64> = epochs_per_sample
      .iter()
      .map(|&eps| eps / negative_sample_rate as f64)
      .collect();

    let epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let epoch_of_next_sample = epochs_per_sample.clone();

    Self {
      manifold,
      head,
      tail,
      epochs_per_sample,
      embedding,
      epoch_of_next_sample,
      epoch_of_next_negative_sample,
      epochs_per_negative_sample,
      current_epoch: 0,
      total_epochs,
      gamma,
      initial_alpha,
      negative_sample_rate: negative_sample_rate as f64,
      metric_type,
    }
  }

  /// Run n more epochs of stochastic gradient descent.
  ///
  /// # Panics
  ///
  /// Panics if this would exceed total_epochs. Check remaining_epochs() first.
  pub fn step_epochs(&mut self, n: usize, output_metric: &dyn Metric) {
    assert!(
      self.current_epoch + n <= self.total_epochs,
      "Cannot step {} epochs: would exceed total_epochs {} (current: {})",
      n,
      self.total_epochs,
      self.current_epoch
    );

    let start_epoch = self.current_epoch;
    let end_epoch = self.current_epoch + n;

    let n_vertices = self.manifold.n_vertices;
    let a = self.manifold.a;
    let b = self.manifold.b;

    // Run the optimization epochs
    let mut embedding_copy = self.embedding.clone();

    for epoch in start_epoch..end_epoch {
      let alpha = self.initial_alpha * (1.0 - (epoch as f32 / self.total_epochs as f32));

      match self.metric_type {
        MetricType::Euclidean => {
          // Euclidean specialization with parallelization
          optimize_layout_euclidean_single_epoch_stateful(
            &mut self.embedding.view_mut(),
            &mut embedding_copy.view_mut(),
            &self.head.view(),
            &self.tail.view(),
            n_vertices,
            &self.epochs_per_sample.view(),
            a,
            b,
            self.gamma,
            alpha,
            &mut self.epochs_per_negative_sample,
            &mut self.epoch_of_next_sample,
            &mut self.epoch_of_next_negative_sample,
            epoch,
            true, // parallel
            true, // move_other
          );
        }
        MetricType::Generic => {
          // Generic metric path
          optimize_layout_generic_single_epoch_stateful(
            &mut self.embedding.view_mut(),
            &mut embedding_copy.view_mut(),
            &self.head.view(),
            &self.tail.view(),
            n_vertices,
            &self.epochs_per_sample.view(),
            a,
            b,
            self.gamma,
            alpha,
            &mut self.epochs_per_negative_sample,
            &mut self.epoch_of_next_sample,
            &mut self.epoch_of_next_negative_sample,
            epoch,
            true, // move_other
            output_metric,
          );
        }
      }
    }

    self.current_epoch = end_epoch;
  }

  /// Get the current epoch number.
  pub fn current_epoch(&self) -> usize {
    self.current_epoch
  }

  /// Get the total epochs this optimizer is configured for.
  pub fn total_epochs(&self) -> usize {
    self.total_epochs
  }

  /// Get the number of remaining epochs.
  pub fn remaining_epochs(&self) -> usize {
    self.total_epochs - self.current_epoch
  }

  /// Get a view of the current embedding.
  pub fn embedding(&self) -> ArrayView2<'_, f32> {
    self.embedding.view()
  }

  /// Get a reference to the learned manifold.
  pub fn manifold(&self) -> &LearnedManifold {
    &self.manifold
  }

  /// Consume the optimizer and return a lightweight fitted model.
  ///
  /// This drops all the optimization state (epoch counters, preprocessed
  /// arrays) and keeps only the manifold and final embedding.
  pub fn into_fitted(self, config: UmapConfig) -> FittedUmap {
    FittedUmap {
      manifold: self.manifold,
      embedding: self.embedding,
      config,
    }
  }
}
