use crate::config::UmapConfig;
use crate::distances::EuclideanMetric;
use crate::manifold::LearnedManifold;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::umap::find_ab_params::find_ab_params;
use crate::umap::fuzzy_simplicial_set::FuzzySimplicialSet;
use crate::umap::raise_disconnected_warning::raise_disconnected_warning;
use dashmap::DashSet;
use ndarray::Array2;
use ndarray::ArrayView2;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::time::Instant;
use tracing::info;

/// UMAP dimensionality reduction algorithm.
///
/// This struct holds the configuration and metrics for UMAP. It can be reused
/// to learn manifolds from multiple datasets with the same parameters.
///
/// # Example
///
/// ```ignore
/// use umap::{Umap, UmapConfig};
/// use ndarray::Array2;
///
/// let config = UmapConfig::default();
/// let umap = Umap::new(config);
///
/// // Learn the manifold structure
/// let manifold = umap.learn_manifold(
///     data.view(),
///     knn_indices.view(),
///     knn_dists.view(),
/// );
///
/// // Create an optimizer and run training
/// let mut opt = Optimizer::new(
///     manifold,
///     init,
///     500, // total epochs
///     config.optimization.repulsion_strength,
///     config.optimization.learning_rate,
///     config.optimization.negative_sample_rate,
///     &euclidean_metric,
/// );
///
/// while opt.remaining_epochs() > 0 {
///     opt.step_epochs(opt.remaining_epochs().min(10));
/// }
///
/// let fitted = opt.into_fitted(config);
/// let embedding = fitted.embedding();
/// ```
pub struct Umap {
  config: UmapConfig,
  metric: Box<dyn Metric>,
  output_metric: Box<dyn Metric>,
}

impl Umap {
  /// Create a new UMAP instance with default Euclidean metrics.
  ///
  /// Both the input space metric (for graph construction) and output space
  /// metric (for optimization) are set to Euclidean distance.
  ///
  /// # Arguments
  ///
  /// * `config` - UMAP configuration parameters
  pub fn new(config: UmapConfig) -> Self {
    Self {
      config,
      metric: Box::new(EuclideanMetric),
      output_metric: Box::new(EuclideanMetric),
    }
  }

  /// Create a UMAP instance with custom distance metrics.
  ///
  /// # Arguments
  ///
  /// * `config` - UMAP configuration parameters
  /// * `metric` - Distance metric for input space (graph construction)
  /// * `output_metric` - Distance metric for output embedding space (optimization)
  ///
  /// # Example
  ///
  /// ```ignore
  /// let umap = Umap::with_metrics(
  ///     config,
  ///     Box::new(MyCustomMetric),
  ///     Box::new(EuclideanMetric),
  /// );
  /// ```
  pub fn with_metrics(
    config: UmapConfig,
    metric: Box<dyn Metric>,
    output_metric: Box<dyn Metric>,
  ) -> Self {
    Self {
      config,
      metric,
      output_metric,
    }
  }

  /// Learn the manifold structure from high-dimensional data.
  ///
  /// This is the expensive graph construction phase that builds a fuzzy
  /// topological representation of the data. The result can be cached,
  /// serialized, and reused for multiple different optimizations.
  ///
  /// This phase is deterministic (no randomness) and independent of the
  /// target embedding dimensionality.
  ///
  /// # Arguments
  ///
  /// * `data` - Input data matrix (n_samples × n_features). Used for validation.
  /// * `knn_indices` - Precomputed k-nearest neighbor indices (n_samples × n_neighbors).
  ///   Each row contains indices of the k nearest neighbors for that sample.
  /// * `knn_dists` - Precomputed k-nearest neighbor distances (n_samples × n_neighbors).
  ///   Each row contains distances to the k nearest neighbors.
  ///
  /// # Returns
  ///
  /// A `LearnedManifold` containing the fuzzy simplicial set and local geometry.
  ///
  /// # Panics
  ///
  /// Panics if:
  /// - Parameter validation fails (invalid ranges, incompatible sizes)
  /// - Array shapes are incompatible
  /// - Number of samples <= n_neighbors
  ///
  /// # Example
  ///
  /// ```ignore
  /// let manifold = umap.learn_manifold(
  ///     data.view(),
  ///     knn_indices.view(),
  ///     knn_dists.view(),
  /// );
  /// // Save for later use
  /// save_manifold(&manifold)?;
  /// ```
  pub fn learn_manifold(
    &self,
    data: ArrayView2<f32>,
    knn_indices: ArrayView2<u32>,
    knn_dists: ArrayView2<f32>,
  ) -> LearnedManifold {
    let n_samples = data.shape()[0];

    // Validate parameters
    self.validate_parameters(n_samples, &knn_indices, &knn_dists);

    // Determine a and b parameters
    let (a, b) =
      if let (Some(a_val), Some(b_val)) = (self.config.manifold.a, self.config.manifold.b) {
        (a_val, b_val)
      } else {
        find_ab_params(self.config.manifold.spread, self.config.manifold.min_dist)
      };

    // Determine disconnection distance
    let disconnection_distance = self
      .config
      .graph
      .disconnection_distance
      .unwrap_or_else(|| self.metric.disconnection_threshold());

    // Find and mark disconnected edges
    let started = Instant::now();
    let knn_disconnections = DashSet::new();
    (0..n_samples).into_par_iter().for_each(|row_no| {
      let row = knn_dists.row(row_no);
      for (col_no, &dist) in row.iter().enumerate() {
        if dist >= disconnection_distance {
          knn_disconnections.insert((row_no, col_no));
        }
      }
    });
    let edges_removed = knn_disconnections.len();
    info!(
      duration_ms = started.elapsed().as_millis(),
      edges_removed, "disconnection detection complete"
    );

    // Build fuzzy simplicial set (the graph)
    info!(
      n_samples,
      n_neighbors = self.config.graph.n_neighbors,
      "starting fuzzy simplicial set"
    );
    let started = Instant::now();
    let (graph, sigmas, rhos) = FuzzySimplicialSet::builder()
      .n_samples(n_samples)
      .n_neighbors(self.config.graph.n_neighbors)
      .knn_indices(knn_indices)
      .knn_dists(knn_dists)
      .knn_disconnections(&knn_disconnections)
      .local_connectivity(self.config.graph.local_connectivity)
      .set_op_mix_ratio(self.config.graph.set_op_mix_ratio)
      .apply_set_operations(true)
      .build()
      .exec();
    info!(
      duration_ms = started.elapsed().as_millis(),
      "fuzzy simplicial set complete"
    );

    // Check for disconnected vertices
    let vertices_disconnected = graph
      .outer_iterator()
      .filter(|row| {
        let sum: f32 = row.data().iter().sum();
        sum == 0.0
      })
      .count();

    raise_disconnected_warning(
      edges_removed,
      vertices_disconnected,
      disconnection_distance,
      n_samples,
      0.1,
    );

    LearnedManifold {
      graph,
      sigmas,
      rhos,
      n_vertices: n_samples,
      a,
      b,
    }
  }

  /// High-level convenience method that learns and optimizes in one call.
  ///
  /// This is equivalent to:
  /// 1. `learn_manifold()` - build the graph
  /// 2. `Optimizer::new()` - set up optimization
  /// 3. Run all epochs
  /// 4. `into_fitted()` - extract final model
  ///
  /// For checkpointing or more control, use the lower-level API instead.
  ///
  /// # Arguments
  ///
  /// * `data` - Input data matrix (n_samples × n_features)
  /// * `knn_indices` - Precomputed k-nearest neighbor indices
  /// * `knn_dists` - Precomputed k-nearest neighbor distances
  /// * `init` - Initial embedding coordinates (n_samples × n_components)
  ///
  /// # Returns
  ///
  /// A `FittedUmap` containing the optimized embedding and learned manifold.
  ///
  /// # Example
  ///
  /// ```ignore
  /// let fitted = umap.fit(
  ///     data.view(),
  ///     knn_indices.view(),
  ///     knn_dists.view(),
  ///     init.view(),
  /// );
  /// let embedding = fitted.embedding();
  /// ```
  pub fn fit(
    &self,
    data: ArrayView2<f32>,
    knn_indices: ArrayView2<u32>,
    knn_dists: ArrayView2<f32>,
    init: ArrayView2<f32>,
  ) -> FittedUmap {
    let n_samples = data.shape()[0];

    // Validate init array
    if init.shape()[1] != self.config.n_components {
      panic!(
        "init has {} components but n_components is {}",
        init.shape()[1],
        self.config.n_components
      );
    }

    if init.shape()[0] != n_samples {
      panic!(
        "init has {} samples but data has {} samples",
        init.shape()[0],
        n_samples
      );
    }

    // Learn the manifold
    let manifold = self.learn_manifold(data, knn_indices, knn_dists);

    // Determine total epochs
    let total_epochs = self
      .config
      .optimization
      .n_epochs
      .unwrap_or_else(|| if n_samples <= 10000 { 500 } else { 200 });

    // Create optimizer
    let metric_type = self.output_metric.metric_type();
    let mut optimizer = Optimizer::new(
      manifold,
      init.to_owned(),
      total_epochs,
      &self.config,
      metric_type,
    );

    // Run all epochs
    optimizer.step_epochs(total_epochs, self.output_metric.as_ref());

    // Extract final model
    let mut fitted = optimizer.into_fitted(self.config.clone());

    // Set disconnected vertices to NaN
    for (i, row) in fitted.manifold.graph.outer_iterator().enumerate() {
      let sum: f32 = row.data().iter().sum();
      if sum == 0.0 {
        for j in 0..fitted.embedding.shape()[1] {
          fitted.embedding[(i, j)] = f32::NAN;
        }
      }
    }

    fitted
  }

  fn validate_parameters(
    &self,
    n_samples: usize,
    knn_indices: &ArrayView2<u32>,
    knn_dists: &ArrayView2<f32>,
  ) {
    // Validate graph parameters
    if self.config.graph.set_op_mix_ratio < 0.0 || self.config.graph.set_op_mix_ratio > 1.0 {
      panic!(
        "set_op_mix_ratio must be between 0.0 and 1.0, got {}",
        self.config.graph.set_op_mix_ratio
      );
    }

    if self.config.graph.n_neighbors < 2 {
      panic!(
        "n_neighbors must be >= 2, got {}",
        self.config.graph.n_neighbors
      );
    }

    // Validate optimization parameters
    if self.config.optimization.repulsion_strength < 0.0 {
      panic!(
        "repulsion_strength cannot be negative, got {}",
        self.config.optimization.repulsion_strength
      );
    }

    if self.config.manifold.min_dist > self.config.manifold.spread {
      panic!(
        "min_dist ({}) must be <= spread ({})",
        self.config.manifold.min_dist, self.config.manifold.spread
      );
    }

    if self.config.manifold.min_dist < 0.0 {
      panic!(
        "min_dist cannot be negative, got {}",
        self.config.manifold.min_dist
      );
    }

    // Validate optimization parameters
    if self.config.optimization.learning_rate < 0.0 {
      panic!(
        "learning_rate must be positive, got {}",
        self.config.optimization.learning_rate
      );
    }

    if self.config.n_components < 1 {
      panic!(
        "n_components must be >= 1, got {}",
        self.config.n_components
      );
    }

    // Validate array shapes
    if knn_dists.shape() != knn_indices.shape() {
      panic!(
        "knn_dists and knn_indices must have the same shape, got {:?} vs {:?}",
        knn_dists.shape(),
        knn_indices.shape()
      );
    }

    if knn_dists.shape()[1] != self.config.graph.n_neighbors {
      panic!(
        "knn_dists has {} neighbors but n_neighbors is {}",
        knn_dists.shape()[1],
        self.config.graph.n_neighbors
      );
    }

    if knn_dists.shape()[0] != n_samples {
      panic!(
        "knn_dists has {} samples but data has {} samples",
        knn_dists.shape()[0],
        n_samples
      );
    }

    // Validate dataset size
    if n_samples <= self.config.graph.n_neighbors {
      panic!(
        "Number of samples ({}) must be > n_neighbors ({})",
        n_samples, self.config.graph.n_neighbors
      );
    }
  }
}

/// A fitted UMAP model containing the learned manifold and embedding.
///
/// This is a lightweight struct that holds only the final results, without
/// the heavy optimization state (epoch counters, preprocessed arrays, etc.).
///
/// The manifold can be serialized and reused for future work like transform().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedUmap {
  pub(crate) embedding: Array2<f32>,
  pub(crate) manifold: LearnedManifold,
  pub(crate) config: UmapConfig,
}

impl FittedUmap {
  /// Get a view of the computed embedding.
  ///
  /// Returns a zero-copy view of the embedding coordinates. Each row
  /// represents one input sample in the low-dimensional space.
  ///
  /// # Returns
  ///
  /// An array view of shape (n_samples, n_components) containing the
  /// embedded coordinates.
  ///
  /// # Example
  ///
  /// ```ignore
  /// let embedding = fitted.embedding();
  /// println!("Embedding shape: {:?}", embedding.shape());
  /// ```
  pub fn embedding(&self) -> ArrayView2<'_, f32> {
    self.embedding.view()
  }

  /// Consume the model and return the embedding, avoiding a copy.
  ///
  /// This method takes ownership of the model and returns the embedding
  /// array directly, which is useful if you don't need the model anymore.
  ///
  /// # Returns
  ///
  /// The embedding array of shape (n_samples, n_components).
  ///
  /// # Example
  ///
  /// ```ignore
  /// let embedding = fitted.into_embedding();
  /// // fitted is now consumed
  /// ```
  pub fn into_embedding(self) -> Array2<f32> {
    self.embedding
  }

  /// Get a reference to the learned manifold.
  pub fn manifold(&self) -> &LearnedManifold {
    &self.manifold
  }

  /// Get a reference to the configuration used for this fit.
  pub fn config(&self) -> &UmapConfig {
    &self.config
  }

  /// Transform new data points into the embedding space.
  ///
  /// **Status: Not yet implemented**
  ///
  /// This method will project new data points into the learned embedding space
  /// using the manifold structure learned during fitting.
  ///
  /// # Arguments
  ///
  /// * `new_data` - New data points to transform (n_new_samples × n_features)
  /// * `new_knn_indices` - KNN indices of new points to training points
  /// * `new_knn_dists` - KNN distances of new points to training points
  ///
  /// # Returns
  ///
  /// Embeddings for the new data points (n_new_samples × n_components)
  ///
  /// # Panics
  ///
  /// Currently panics with "not yet implemented" message.
  #[allow(unused_variables)]
  pub fn transform(
    &self,
    new_data: ArrayView2<f32>,
    new_knn_indices: ArrayView2<u32>,
    new_knn_dists: ArrayView2<f32>,
  ) -> Array2<f32> {
    todo!("Transform not yet implemented - contributions welcome!")
  }
}
