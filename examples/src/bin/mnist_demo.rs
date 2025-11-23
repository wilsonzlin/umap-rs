use clap::Parser;
use clap::ValueEnum;
use mnist::Mnist;
use mnist::MnistBuilder;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO; // Used for PCA
use plotters::prelude::*;
use rand::Rng;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use umap_rs::Umap;
use umap_rs::UmapConfig;

#[derive(Parser)]
#[command(name = "MNIST UMAP Demo")]
struct Args {
  /// Initialization method
  #[arg(short, long, default_value = "random")]
  init: InitMethod,

  /// Number of samples to use (max 60000)
  #[arg(short, long, default_value = "60000")]
  samples: usize,

  /// Output PNG path
  #[arg(short, long, default_value = "mnist_umap.png")]
  output: String,

  /// Number of optimization epochs (default: auto-determined)
  #[arg(short, long)]
  epochs: Option<usize>,
}

#[derive(ValueEnum, Clone, Debug)]
enum InitMethod {
  Random,
  Pca,
}

fn main() {
  let args = Args::parse();

  println!("MNIST UMAP Demo");
  println!("================");
  println!("Initialization: {:?}", args.init);
  println!("Samples: {}", args.samples);
  println!();

  // Load MNIST
  println!("Loading MNIST dataset...");
  let start = Instant::now();
  let (data, labels) = load_mnist(args.samples);
  println!(
    "  Loaded {} samples in {:.2}s",
    data.shape()[0],
    start.elapsed().as_secs_f32()
  );
  println!();

  // Compute KNN
  println!("Computing k-nearest neighbors (k=15)...");
  let start = Instant::now();
  let (knn_indices, knn_dists) = compute_knn(data.view(), 15);
  println!("  KNN computed in {:.2}s", start.elapsed().as_secs_f32());
  println!();

  // Compute initialization
  println!("Computing {:?} initialization...", args.init);
  let start = Instant::now();
  let init = match args.init {
    InitMethod::Random => compute_random_init(data.shape()[0]),
    InitMethod::Pca => compute_pca_init(data.view()),
  };
  println!(
    "  Initialization computed in {:.2}s",
    start.elapsed().as_secs_f32()
  );
  println!();

  // Run UMAP
  println!("Running UMAP optimization...");
  let start = Instant::now();
  let config = UmapConfig {
    optimization: umap_rs::OptimizationParams {
      n_epochs: args.epochs,
      ..Default::default()
    },
    ..Default::default()
  };
  let umap = Umap::new(config);
  let model = umap.fit(
    data.view(),
    knn_indices.view(),
    knn_dists.view(),
    init.view(),
  );
  let embedding = model.into_embedding();
  println!("  UMAP completed in {:.2}s", start.elapsed().as_secs_f32());
  println!();

  // Generate plot
  println!("Generating scatter plot...");
  let start = Instant::now();
  plot_embedding(&embedding, &labels, &args.output, &args.init).unwrap();
  println!(
    "  Plot saved to {} in {:.2}s",
    args.output,
    start.elapsed().as_secs_f32()
  );
  println!();
  println!("Done!");
}

/// Load MNIST dataset and normalize to [0, 1]
fn load_mnist(n_samples: usize) -> (Array2<f32>, Vec<u8>) {
  let n_samples = n_samples.min(60000);

  // Download MNIST if not present
  download_mnist_if_needed();

  let Mnist {
    trn_img, trn_lbl, ..
  } = MnistBuilder::new()
    .label_format_digit()
    .training_set_length(60000)
    .test_set_length(0)
    .base_path("data/")
    .finalize();

  // Convert to f32 and normalize
  let data = Array2::from_shape_vec((60000, 28 * 28), trn_img)
    .unwrap()
    .mapv(|x| x as f32 / 255.0);

  // Subsample if needed
  let data = data.slice(s![0..n_samples, ..]).to_owned();
  let labels = trn_lbl[0..n_samples].to_vec();

  (data, labels)
}

/// Download MNIST dataset if not already present
fn download_mnist_if_needed() {
  let data_dir = Path::new("data");
  let files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
  ];

  let all_present = files.iter().all(|f| data_dir.join(f).exists());

  if all_present {
    return;
  }

  println!("Downloading MNIST dataset...");
  fs::create_dir_all(data_dir).unwrap();

  let base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
  let files_with_ext = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
  ];

  for file in &files_with_ext {
    let uncompressed_path = data_dir.join(file.trim_end_matches(".gz"));

    if uncompressed_path.exists() {
      continue;
    }

    println!("  Downloading {}...", file);
    let url = format!("{}{}", base_url, file);
    let response = ureq::get(&url).call().unwrap();
    let compressed = response.into_body().read_to_vec().unwrap();

    println!("  Decompressing {}...", file);
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut decompressed).unwrap();

    let mut output = fs::File::create(&uncompressed_path).unwrap();
    output.write_all(&decompressed).unwrap();
  }

  println!("  MNIST dataset ready!");
  println!();
}

/// Compute k-nearest neighbors using brute-force Euclidean distance
fn compute_knn(data: ArrayView2<f32>, k: usize) -> (Array2<u32>, Array2<f32>) {
  let n_samples = data.shape()[0];
  let mut knn_indices = Array2::zeros((n_samples, k));
  let mut knn_dists = Array2::zeros((n_samples, k));

  for i in 0..n_samples {
    let point = data.row(i);
    let mut distances: Vec<(usize, f32)> = (0..n_samples)
      .filter(|&j| i != j)
      .map(|j| {
        let other = data.row(j);
        let dist = euclidean_distance(point, other);
        (j, dist)
      })
      .collect();

    // Sort by distance and take k nearest
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (k_idx, &(j, dist)) in distances.iter().take(k).enumerate() {
      knn_indices[(i, k_idx)] = j as u32;
      knn_dists[(i, k_idx)] = dist;
    }
  }

  (knn_indices, knn_dists)
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).powi(2))
    .sum::<f32>()
    .sqrt()
}

/// Random initialization: uniform distribution in [-10, 10]
fn compute_random_init(n_samples: usize) -> Array2<f32> {
  let mut rng = rand::rng();
  Array2::from_shape_fn((n_samples, 2), |_| rng.random_range(-10.0..10.0))
}

/// PCA initialization: project onto top 2 principal components
fn compute_pca_init(data: ArrayView2<f32>) -> Array2<f32> {
  let n_samples = data.shape()[0];

  // Center the data
  let mean = data.mean_axis(Axis(0)).unwrap();
  let centered = &data - &mean.insert_axis(Axis(0));

  // Compute covariance matrix: C = (1/n) * X^T * X
  let cov = centered.t().dot(&centered) / (n_samples as f32);

  // Eigendecomposition (for symmetric matrix, use eigh)
  let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper).unwrap();

  // Sort eigenvalues and eigenvectors in descending order
  let mut eigen_pairs: Vec<(f32, Array1<f32>)> = eigenvalues
    .iter()
    .zip(eigenvectors.axis_iter(Axis(1)))
    .map(|(val, vec)| (*val, vec.to_owned()))
    .collect();

  eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

  // Take top 2 eigenvectors
  let pc1 = &eigen_pairs[0].1;
  let pc2 = &eigen_pairs[1].1;

  // Project data onto principal components
  let mut projection = Array2::zeros((n_samples, 2));
  for i in 0..n_samples {
    let point = centered.row(i);
    projection[(i, 0)] = point.dot(pc1);
    projection[(i, 1)] = point.dot(pc2);
  }

  // Scale to reasonable range [-10, 10]
  scale_to_range(&mut projection, -10.0, 10.0);

  projection
}

/// Scale array to specified range
fn scale_to_range(arr: &mut Array2<f32>, min_val: f32, max_val: f32) {
  for col_idx in 0..arr.shape()[1] {
    let mut col = arr.column_mut(col_idx);
    let current_min = col.iter().cloned().fold(f32::INFINITY, f32::min);
    let current_max = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let current_range = current_max - current_min;

    if current_range > 0.0 {
      let target_range = max_val - min_val;
      for val in col.iter_mut() {
        *val = min_val + (*val - current_min) * target_range / current_range;
      }
    }
  }
}

/// Generate scatter plot colored by digit labels
fn plot_embedding(
  embedding: &Array2<f32>,
  labels: &[u8],
  output_path: &str,
  init_method: &InitMethod,
) -> Result<(), Box<dyn std::error::Error>> {
  let root = BitMapBackend::new(output_path, (1024, 1024)).into_drawing_area();
  root.fill(&WHITE)?;

  // Compute axis ranges
  let x_vals: Vec<f32> = embedding.column(0).iter().cloned().collect();
  let y_vals: Vec<f32> = embedding.column(1).iter().cloned().collect();

  let x_min = x_vals.iter().cloned().fold(f32::INFINITY, f32::min);
  let x_max = x_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let y_min = y_vals.iter().cloned().fold(f32::INFINITY, f32::min);
  let y_max = y_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

  // Add padding
  let x_padding = (x_max - x_min) * 0.05;
  let y_padding = (y_max - y_min) * 0.05;

  let mut chart = ChartBuilder::on(&root)
    .caption(
      format!("MNIST UMAP - {:?} Initialization", init_method),
      ("sans-serif", 40).into_font(),
    )
    .margin(10)
    .x_label_area_size(40)
    .y_label_area_size(50)
    .build_cartesian_2d(
      x_min - x_padding..x_max + x_padding,
      y_min - y_padding..y_max + y_padding,
    )?;

  chart
    .configure_mesh()
    .x_desc("UMAP 1")
    .y_desc("UMAP 2")
    .draw()?;

  // Define 10 distinct colors for digits 0-9
  let colors = [
    RGBColor(228, 26, 28),   // Red
    RGBColor(55, 126, 184),  // Blue
    RGBColor(77, 175, 74),   // Green
    RGBColor(152, 78, 163),  // Purple
    RGBColor(255, 127, 0),   // Orange
    RGBColor(255, 255, 51),  // Yellow
    RGBColor(166, 86, 40),   // Brown
    RGBColor(247, 129, 191), // Pink
    RGBColor(153, 153, 153), // Gray
    RGBColor(0, 0, 0),       // Black
  ];

  // Draw points
  for i in 0..embedding.shape()[0] {
    let x = embedding[(i, 0)];
    let y = embedding[(i, 1)];
    let label = labels[i] as usize;
    let color = colors[label % 10];

    chart.draw_series(std::iter::once(Circle::new((x, y), 2, color.filled())))?;
  }

  // Draw legend
  for (i, &color) in colors.iter().enumerate() {
    chart
      .draw_series(std::iter::once(Circle::new((0.0, 0.0), 0, color.filled())))?
      .label(format!("Digit {}", i))
      .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
  }

  chart
    .configure_series_labels()
    .border_style(&BLACK)
    .background_style(&WHITE.mix(0.8))
    .position(SeriesLabelPosition::UpperRight)
    .draw()?;

  root.present()?;
  Ok(())
}
