use crate::config::UmapConfig;
use crate::embedding::FittedUmap;
use crate::layout::optimize_layout_euclidean::optimize_layout_euclidean_single_epoch_stateful;
use crate::layout::optimize_layout_generic::optimize_layout_generic_single_epoch_stateful;
use crate::manifold::LearnedManifold;
use crate::metric::Metric;
use crate::metric::MetricType;
use crate::umap::make_epochs_per_sample::make_epochs_per_sample;
use crate::utils::parallel_vec::ParallelVec;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::time::Instant;
use tracing::info;

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
    let started = Instant::now();
    let max_val = graph
      .data()
      .par_iter()
      .copied()
      .reduce(|| 0.0f32, |a, b| a.max(b));

    let default_epochs = if n_samples <= 10000 { 500 } else { 200 };
    let threshold_epochs = if total_epochs > 10 {
      total_epochs
    } else {
      default_epochs
    };
    let threshold = max_val / threshold_epochs as f32;
    info!(
      duration_ms = started.elapsed().as_millis(),
      max_val, threshold, "optimizer threshold computed"
    );

    // Count edges per row that pass threshold (parallel)
    let started = Instant::now();
    let row_counts: Vec<usize> = (0..n_samples)
      .into_par_iter()
      .map(|row| {
        let row_start = graph.indptr().index(row);
        let row_end = graph.indptr().index(row + 1);
        let row_data = &graph.data()[row_start..row_end];
        row_data.iter().filter(|&&v| v >= threshold).count()
      })
      .collect();

    // Prefix sum for edge offsets
    let mut edge_offsets: Vec<usize> = Vec::with_capacity(n_samples + 1);
    edge_offsets.push(0);
    let mut total_edges = 0usize;
    for &count in &row_counts {
      total_edges += count;
      edge_offsets.push(total_edges);
    }
    info!(
      duration_ms = started.elapsed().as_millis(),
      total_edges, "optimizer edge filtering complete"
    );

    // Extract head, tail, and weights in parallel
    let started = Instant::now();
    let head_vec = ParallelVec::new(vec![0u32; total_edges]);
    let tail_vec = ParallelVec::new(vec![0u32; total_edges]);
    let weights_vec = ParallelVec::new(vec![0.0f32; total_edges]);

    (0..n_samples).into_par_iter().for_each(|row| {
      let row_start = graph.indptr().index(row);
      let row_end = graph.indptr().index(row + 1);
      let row_indices = &graph.indices()[row_start..row_end];
      let row_data = &graph.data()[row_start..row_end];

      let out_start = edge_offsets[row];
      let mut offset = 0;

      for (&col, &val) in row_indices.iter().zip(row_data) {
        if val >= threshold {
          // SAFETY: Each row writes to disjoint section [edge_offsets[row]..edge_offsets[row+1]]
          unsafe {
            head_vec.write(out_start + offset, row as u32);
            tail_vec.write(out_start + offset, col);
            weights_vec.write(out_start + offset, val);
          }
          offset += 1;
        }
      }
    });

    let head = head_vec.into_inner();
    let tail = tail_vec.into_inner();
    let weights = weights_vec.into_inner();
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer edge extraction complete"
    );

    // Compute epochs per sample from edge weights
    let started = Instant::now();
    let weights_array = Array1::from(weights);
    let epochs_per_sample = make_epochs_per_sample(&weights_array.view(), total_epochs);

    let head = Array1::from(head);
    let tail = Array1::from(tail);
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer epochs_per_sample complete"
    );

    // Normalize embedding to [0, 10] range (parallel per column)
    let started = Instant::now();
    let mut embedding = init;
    let n_cols = embedding.shape()[1];
    for col_idx in 0..n_cols {
      let col_slice = embedding.column(col_idx);
      let col_data = col_slice.as_slice().unwrap();

      // Parallel min/max
      let (min, max) = col_data
        .par_iter()
        .copied()
        .fold(
          || (f32::INFINITY, f32::NEG_INFINITY),
          |(min, max), v| (min.min(v), max.max(v)),
        )
        .reduce(
          || (f32::INFINITY, f32::NEG_INFINITY),
          |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
        );

      let range = max - min;
      if range > 0.0 {
        let scale = 10.0 / range;
        embedding
          .column_mut(col_idx)
          .as_slice_mut()
          .unwrap()
          .par_iter_mut()
          .for_each(|v| *v = (*v - min) * scale);
      }
    }
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer embedding normalization complete"
    );

    // Initialize epoch scheduling (one at a time to avoid memory spike).
    // No perf loss: each array is still computed in parallel, just not allocated simultaneously.
    let started = Instant::now();
    let neg_rate = negative_sample_rate as f64;
    let eps_slice = epochs_per_sample.as_slice().unwrap();

    let epoch_of_next_sample = Array1::from(
      eps_slice.par_iter().copied().collect::<Vec<_>>(),
    );
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer epoch_of_next_sample complete"
    );

    let started = Instant::now();
    let epochs_per_negative_sample = Array1::from(
      eps_slice
        .par_iter()
        .map(|&eps| eps / neg_rate)
        .collect::<Vec<_>>(),
    );
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer epochs_per_negative_sample complete"
    );

    let started = Instant::now();
    let epoch_of_next_negative_sample = Array1::from(
      epochs_per_negative_sample
        .as_slice()
        .unwrap()
        .par_iter()
        .copied()
        .collect::<Vec<_>>(),
    );
    info!(
      duration_ms = started.elapsed().as_millis(),
      "optimizer epoch_of_next_negative_sample complete"
    );

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
