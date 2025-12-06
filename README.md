# Signal Processing and Filtering

An exploration tutorial on signal processing, filtering, and kernel-based image processing, progressing from fundamentals to GPU-accelerated implementations.

## Overview

This repository provides a hands-on learning path through signal processing and filtering concepts, starting with basic 1D signals and progressing to high-performance GPU-based 2D image filtering. Each module includes theory, implementations, visualizations, and practical examples.

## Learning Path

### Phase 1: Fundamentals
1. **[Signals and Systems](01-signals-and-systems/)** - Continuous and discrete signals, impulse response, LTI systems
2. **[1D Convolution](02-1d-convolution/)** - Mathematical foundation, implementation from scratch, practical examples
3. **[Digital Signals](03-digital-signals/)** - Sampling theorem, quantization, DFT/FFT analysis

### Phase 2: Analysis Tools
4. **[Correlation](04-correlation/)** - Auto-correlation, cross-correlation, pattern matching applications

### Phase 3: Image Processing
5. **[2D Signals and Images](05-2d-signals-images/)** - Images as 2D signals, kernel representation
6. **[2D Convolution and Image Filtering](06-2d-convolution/)** - Extending convolution to 2D, basic image filters
7. **[Advanced Filters](07-advanced-filters/)** - Gaussian blur, edge detection (Sobel, Laplacian), separable filters

### Phase 4: Performance Optimization
8. **[GPU Acceleration Basics](08-gpu-basics/)** - Parallel computing concepts, CUDA/OpenCL introduction
9. **[GPU Image Filtering](09-gpu-image-filtering/)** - High-performance kernel-based filtering on GPU

## Prerequisites

- Python 3.8+
- Basic understanding of linear algebra
- Familiarity with NumPy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd filtering

# Install dependencies
pip install -r requirements.txt
```

## Usage

Each module is self-contained with:
- Theory documentation (README.md)
- Python implementations
- Jupyter notebooks with examples
- Exercises and solutions

Start with Module 1 and progress sequentially for the best learning experience.

## Project Structure

```
filtering/
├── 01-signals-and-systems/
├── 02-1d-convolution/
├── 03-digital-signals/
├── 04-correlation/
├── 05-2d-signals-images/
├── 06-2d-convolution/
├── 07-advanced-filters/
├── 08-gpu-basics/
├── 09-gpu-image-filtering/
├── assets/              # Images and resources
├── utils/               # Shared utility functions
└── requirements.txt
```

## Contributing

This is an educational project. Suggestions and improvements are welcome!

## License

MIT License
