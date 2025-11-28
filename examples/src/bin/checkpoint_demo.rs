/// Demonstration of checkpoint/resume functionality in UMAP.
///
/// This example shows how to:
/// 1. Learn a manifold from data
/// 2. Create an optimizer
/// 3. Train with checkpoints
/// 4. Save/load checkpoints for fault tolerance
/// 5. Resume training from a checkpoint
use ndarray::Array2;
use rand::Rng;
use std::fs;
use std::io::Read;
use umap_rs::metric::Metric;
use umap_rs::EuclideanMetric;
use umap_rs::Optimizer;
use umap_rs::Umap;
use umap_rs::UmapConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("UMAP Checkpoint/Resume Demo");
  println!();

  // Generate synthetic data (100 samples, 50 dimensions)
  println!("Generating synthetic data...");
  let n_samples = 100;
  let n_features = 50;
  let n_neighbors = 15;

  let mut rng = rand::rng();
  let data: Array2<f32> = Array2::from_shape_fn((n_samples, n_features), |_| rng.random());

  // Generate fake KNN data (normally you'd use a real KNN library)
  let mut knn_indices = Array2::<u32>::zeros((n_samples, n_neighbors));
  let mut knn_dists = Array2::<f32>::zeros((n_samples, n_neighbors));

  for i in 0..n_samples {
    for j in 0..n_neighbors {
      // Just use sequential indices as fake neighbors
      knn_indices[(i, j)] = ((i + j + 1) % n_samples) as u32;
      knn_dists[(i, j)] = rng.random::<f32>() * 2.0;
    }
  }

  // Initialize embedding with random coordinates
  let init: Array2<f32> = Array2::from_shape_fn((n_samples, 2), |_| rng.random::<f32>() * 10.0);

  // Create UMAP configuration
  let config = UmapConfig::default();
  let umap = Umap::new(config.clone());

  // Phase 1: Learn the manifold (expensive, cacheable)
  println!();
  println!("=== Phase 1: Learning manifold structure ===");
  let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

  println!("Manifold learned:");
  println!("  - n_vertices: {}", manifold.n_vertices());
  println!(
    "  - curve params: ({:.4}, {:.4})",
    manifold.curve_params().0,
    manifold.curve_params().1
  );
  println!("  - graph edges: {}", manifold.graph().nnz());

  // Save the manifold (you could load this later to skip Phase 1)
  println!();
  println!("Saving manifold to disk...");
  let manifold_bytes = bincode::serialize(&manifold)?;
  fs::write("checkpoint_manifold.bin", &manifold_bytes)?;
  println!("  Manifold saved ({} bytes)", manifold_bytes.len());

  // Phase 2: Create optimizer and train with checkpoints
  println!();
  println!("=== Phase 2: Optimization with checkpoints ===");
  let total_epochs = 100;

  let metric = EuclideanMetric;
  let metric_type = metric.metric_type();

  let mut opt = Optimizer::new(manifold.clone(), init, total_epochs, &config, metric_type);

  println!("Optimizer created for {} epochs", total_epochs);

  // Train in chunks with checkpoints
  let checkpoint_interval = 10;

  println!();
  println!(
    "Training with checkpoints every {} epochs:",
    checkpoint_interval
  );

  while opt.remaining_epochs() > 0 {
    let epochs_to_run = opt.remaining_epochs().min(checkpoint_interval);
    opt.step_epochs(epochs_to_run, &metric);

    println!(
      "  Epoch {}/{} - embedding mean: {:.4}",
      opt.current_epoch(),
      opt.total_epochs(),
      opt.embedding().mean().unwrap()
    );

    // Save checkpoint every interval
    if opt.current_epoch() % checkpoint_interval == 0 {
      let checkpoint_bytes = bincode::serialize(&opt)?;
      let filename = format!("checkpoint_{:03}.bin", opt.current_epoch());
      fs::write(&filename, &checkpoint_bytes)?;
      println!(
        "    ✓ Checkpoint saved: {} ({} bytes)",
        filename,
        checkpoint_bytes.len()
      );
    }
  }

  println!();
  println!("=== Phase 3: Converting to final model ===");
  let fitted = opt.into_fitted(config.clone());
  println!(
    "Training complete! Final embedding shape: {:?}",
    fitted.embedding().shape()
  );

  // Demonstrate resuming from a checkpoint
  println!();
  println!("=== Demo: Resuming from checkpoint ===");
  println!("Loading checkpoint from epoch 50...");

  let mut checkpoint_file = fs::File::open("checkpoint_050.bin")?;
  let mut checkpoint_bytes = Vec::new();
  checkpoint_file.read_to_end(&mut checkpoint_bytes)?;

  let mut resumed_opt: Optimizer = bincode::deserialize(&checkpoint_bytes)?;

  println!(
    "Checkpoint loaded! Currently at epoch {}/{}",
    resumed_opt.current_epoch(),
    resumed_opt.total_epochs()
  );
  println!("Remaining epochs: {}", resumed_opt.remaining_epochs());

  // Continue training from checkpoint
  println!();
  println!("Continuing training from checkpoint...");
  while resumed_opt.remaining_epochs() > 0 {
    let epochs_to_run = resumed_opt.remaining_epochs().min(checkpoint_interval);
    resumed_opt.step_epochs(epochs_to_run, &metric);

    println!(
      "  Epoch {}/{} - embedding mean: {:.4}",
      resumed_opt.current_epoch(),
      resumed_opt.total_epochs(),
      resumed_opt.embedding().mean().unwrap()
    );
  }

  let resumed_fitted = resumed_opt.into_fitted(config);
  println!(
    "\nResumed training complete! Final embedding shape: {:?}",
    resumed_fitted.embedding().shape()
  );

  // Cleanup checkpoint files
  println!();
  println!("=== Cleanup ===");
  for i in (10..=100).step_by(10) {
    let filename = format!("checkpoint_{:03}.bin", i);
    if fs::remove_file(&filename).is_ok() {
      println!("Removed {}", filename);
    }
  }
  fs::remove_file("checkpoint_manifold.bin")?;
  println!("Cleanup complete!");

  Ok(())
}
