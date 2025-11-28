#[cfg(test)]
mod tests {
  use crate::FittedUmap;
  use crate::LearnedManifold;
  use crate::Optimizer;
  use crate::Umap;
  use crate::UmapConfig;
  use crate::distances::EuclideanMetric;
  use crate::metric::Metric;
  use ndarray::Array2;
  use rand::Rng;

  /// Generate synthetic test data
  fn generate_test_data() -> (Array2<f32>, Array2<u32>, Array2<f32>, Array2<f32>) {
    let n_samples = 50;
    let n_features = 20;
    let n_neighbors = 10;
    let n_components = 2;

    let mut rng = rand::rng();

    let data: Array2<f32> = Array2::from_shape_fn((n_samples, n_features), |_| rng.random());

    let mut knn_indices = Array2::<u32>::zeros((n_samples, n_neighbors));
    let mut knn_dists = Array2::<f32>::zeros((n_samples, n_neighbors));

    for i in 0..n_samples {
      for j in 0..n_neighbors {
        knn_indices[(i, j)] = ((i + j + 1) % n_samples) as u32;
        knn_dists[(i, j)] = rng.random::<f32>() * 2.0;
      }
    }

    let init: Array2<f32> =
      Array2::from_shape_fn((n_samples, n_components), |_| rng.random::<f32>() * 10.0);

    (data, knn_indices, knn_dists, init)
  }

  #[test]
  fn test_high_level_fit() {
    let (data, knn_indices, knn_dists, init) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;
    config.optimization.n_epochs = Some(10);

    let umap = Umap::new(config);
    let fitted = umap.fit(
      data.view(),
      knn_indices.view(),
      knn_dists.view(),
      init.view(),
    );

    assert_eq!(fitted.embedding().shape(), &[50, 2]);
    assert!(
      fitted
        .embedding()
        .iter()
        .all(|&x| x.is_finite() || x.is_nan())
    );
  }

  #[test]
  fn test_learn_manifold() {
    let (data, knn_indices, knn_dists, _) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;

    let umap = Umap::new(config);
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    assert_eq!(manifold.n_vertices(), 50);
    assert!(manifold.graph().nnz() > 0);
    let (a, b) = manifold.curve_params();
    assert!(a > 0.0 && b > 0.0);
  }

  #[test]
  fn test_optimizer_step_epochs() {
    let (data, knn_indices, knn_dists, init) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;

    let umap = Umap::new(config.clone());
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    let total_epochs = 20;
    let metric = EuclideanMetric;
    let metric_type = metric.metric_type();

    let mut opt = Optimizer::new(manifold, init, total_epochs, &config, metric_type);

    assert_eq!(opt.current_epoch(), 0);
    assert_eq!(opt.remaining_epochs(), 20);

    opt.step_epochs(5, &metric);
    assert_eq!(opt.current_epoch(), 5);
    assert_eq!(opt.remaining_epochs(), 15);

    opt.step_epochs(15, &metric);
    assert_eq!(opt.current_epoch(), 20);
    assert_eq!(opt.remaining_epochs(), 0);

    let fitted = opt.into_fitted(config);
    assert_eq!(fitted.embedding().shape(), &[50, 2]);
  }

  #[test]
  #[should_panic(expected = "Cannot step 5 epochs: would exceed total_epochs 20 (current: 18)")]
  fn test_optimizer_overstep_panics() {
    let (data, knn_indices, knn_dists, init) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;

    let umap = Umap::new(config.clone());
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    let metric = EuclideanMetric;
    let metric_type = metric.metric_type();

    let mut opt = Optimizer::new(manifold, init, 20, &config, metric_type);

    opt.step_epochs(18, &metric);
    opt.step_epochs(5, &metric); // Should panic: 18 + 5 > 20
  }

  #[test]
  fn test_checkpoint_serialization() {
    let (data, knn_indices, knn_dists, init) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;

    let umap = Umap::new(config.clone());
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    let metric = EuclideanMetric;
    let metric_type = metric.metric_type();

    let mut opt = Optimizer::new(manifold, init, 50, &config, metric_type);

    opt.step_epochs(20, &metric);

    // Serialize
    let serialized = bincode::serialize(&opt).expect("Serialization failed");

    // Deserialize
    let mut opt2: Optimizer = bincode::deserialize(&serialized).expect("Deserialization failed");

    assert_eq!(opt2.current_epoch(), 20);
    assert_eq!(opt2.remaining_epochs(), 30);

    // Continue training
    opt2.step_epochs(30, &metric);
    let fitted = opt2.into_fitted(config);
    assert_eq!(fitted.embedding().shape(), &[50, 2]);
  }

  #[test]
  fn test_manifold_serialization() {
    let (data, knn_indices, knn_dists, _) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;

    let umap = Umap::new(config);
    let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

    let (orig_a, orig_b) = manifold.curve_params();
    let orig_vertices = manifold.n_vertices();
    let orig_edges = manifold.graph().nnz();

    // Serialize and deserialize
    let serialized = bincode::serialize(&manifold).expect("Serialization failed");
    let manifold2: LearnedManifold =
      bincode::deserialize(&serialized).expect("Deserialization failed");

    let (new_a, new_b) = manifold2.curve_params();
    assert_eq!(new_a, orig_a);
    assert_eq!(new_b, orig_b);
    assert_eq!(manifold2.n_vertices(), orig_vertices);
    assert_eq!(manifold2.graph().nnz(), orig_edges);
  }

  #[test]
  fn test_fitted_umap_serialization() {
    let (data, knn_indices, knn_dists, init) = generate_test_data();

    let mut config = UmapConfig::default();
    config.graph.n_neighbors = 10;
    config.optimization.n_epochs = Some(10);

    let umap = Umap::new(config);
    let fitted = umap.fit(
      data.view(),
      knn_indices.view(),
      knn_dists.view(),
      init.view(),
    );

    // Serialize and deserialize
    let serialized = bincode::serialize(&fitted).expect("Serialization failed");
    let fitted2: FittedUmap = bincode::deserialize(&serialized).expect("Deserialization failed");

    assert_eq!(fitted.embedding().shape(), fitted2.embedding().shape());
    assert_eq!(
      fitted.manifold().n_vertices(),
      fitted2.manifold().n_vertices()
    );
  }
}
