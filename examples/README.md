# MNIST UMAP Demo

This example demonstrates UMAP dimensionality reduction on the MNIST handwritten digits dataset.

## Features

- Automatic MNIST dataset download (first run only)
- Three initialization methods: Random, PCA, and Spectral
- K-nearest neighbors computation (brute-force)
- PNG scatter plot output with digit labels color-coded

## Usage

Run the example with:

```bash
cargo run --example mnist_demo --features "clap,mnist,ndarray-linalg,plotters,ureq,flate2" --release -- [OPTIONS]
```

On first run, the MNIST dataset will be automatically downloaded to the `data/` directory.

### Options

- `--init <METHOD>` - Initialization method: `random`, `pca`, or `spectral` (default: `random`)
- `--samples <N>` - Number of samples to use, max 60000 (default: `60000`)
- `--output <PATH>` - Output PNG file path (default: `mnist_umap.png`)
- `--epochs <N>` - Number of optimization epochs (default: auto-determined based on dataset size)

## Implementation Details

### Initialization Methods

1. **Random**: Uniform distribution in [-10, 10]
   - Simplest approach, always works
   - Good baseline for comparison

2. **PCA**: Project data onto top 2 principal components
   - **Recommended** for better convergence
   - Faster and more reliable than spectral

3. **Spectral**: Graph Laplacian eigenvectors of the k-NN graph
   - ⚠️ **TOY IMPLEMENTATION** - Demo purposes only
   - Uses dense eigendecomposition (O(n³))
   - Impractical for >5,000 samples
   - Rebuilds graph instead of using fuzzy simplicial set
   - **Use Random or PCA instead for production code**

### Visualization

- Scatter plot with 10 distinct colors for digits 0-9
- Auto-scaled axes based on embedding ranges
- Legend showing digit-to-color mapping
