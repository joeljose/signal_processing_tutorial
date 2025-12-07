# Signal Processing and Filtering

An exploration tutorial on signal processing, filtering, and kernel-based image processing, progressing from fundamentals to GPU-accelerated implementations.

Based on Oppenheim & Schafer's "Discrete-Time Signal Processing"

## Overview

This repository provides a hands-on learning path through signal processing and filtering concepts, starting with basic 1D signals and progressing to high-performance GPU-based 2D image filtering. Each module includes theory, implementations, visualizations, and practical examples.

## Learning Path

### Phase 1: Time Domain Fundamentals

1. **[Signals and Systems](01-signals-and-systems/)** - Continuous and discrete signals, impulse response, LTI systems, periodicity
2. **[1D Convolution](02-1d-convolution/)** - Mathematical foundation, direct and FFT-based implementation, filtering applications


### Phase 2: Frequency Domain Analysis

3. **[DTFT and Sampling](03-dtft-sampling/)** - Discrete-Time Fourier Transform, sampling theorem, aliasing, Nyquist criterion
4. **[DFT and FFT](04-dft-fft/)** - Discrete Fourier Transform, FFT algorithm, frequency bins, computational aspects
5. **[Frequency Domain Filtering](05-frequency-filtering/)** - Convolution theorem, spectral analysis, filter design in frequency domain
6. **[Windowing and Spectral Analysis](06-windowing/)** - Window functions, spectral leakage, zero-padding effects

### Phase 3: Correlation and Analysis Tools

7. **[Correlation](07-correlation/)** - Auto-correlation, cross-correlation, pattern matching applications

### Phase 4: 2D Signal Processing and Images

8. **[2D Signals and Images](08-2d-signals-images/)** - Images as 2D signals, 2D DTFT, 2D DFT, separable transforms
9. **[2D DFT and Image Filtering](09-2d-dft-filtering/)** - 2D convolution, frequency domain filtering, separable filters
10. **[Advanced Image Filters](10-advanced-filters/)** - Gaussian blur, edge detection (Sobel, Laplacian, Canny), morphological operations

### Phase 5: High-Performance Computing

11. **[GPU Acceleration Basics](11-gpu-acceleration/)** - Parallel computing concepts, GPU programming frameworks (CUDA, CuPy, Triton), memory management
12. **[NVIDIA Libraries](12-nvidia-libraries/)** - Production-grade GPU libraries: cuFFT (FFT), NPP (image processing), Thrust (parallel algorithms)
13. **[GPU Image Filtering](13-gpu-image-filtering/)** - Practical GPU-accelerated filtering implementations using multiple frameworks

## Key Concepts Covered

### Fourier Transform Family

Understanding the relationships between different Fourier transforms:

- **CTFT** (Continuous-Time FT): Continuous aperiodic signals
- **FS** (Fourier Series): Continuous periodic signals
- **DTFT** (Discrete-Time FT): Discrete aperiodic signals
- **DFT** (Discrete FT): Discrete periodic/finite-length signals

### Duality Principle

> **Periodicity in one domain ⟺ Discreteness in the other domain**

| Transform | Time Domain | Frequency Domain |
|-----------|-------------|------------------|
| CTFT | Continuous, Aperiodic | Continuous, Aperiodic |
| FS | Continuous, Periodic | Discrete, Aperiodic |
| DTFT | Discrete, Aperiodic | Continuous, Periodic |
| DFT | Discrete, Periodic | Discrete, Periodic |

## Prerequisites

- Python 3.8+
- Basic understanding of linear algebra
- Familiarity with NumPy
- Understanding of complex numbers

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

- Theory documentation (README.md) with LaTeX equations
- Python implementations
- Jupyter notebooks with interactive examples
- Exercises and solutions
- Mermaid diagrams for visualization

Start with Module 1 and progress sequentially for the best learning experience.

## Project Structure

```
filtering/
├── 01-signals-and-systems/
├── 02-1d-convolution/
├── 03-dtft-sampling/
├── 04-dft-fft/
├── 05-frequency-filtering/
├── 06-windowing/
├── 07-correlation/
├── 08-2d-signals-images/
├── 09-2d-dft-filtering/
├── 10-advanced-filters/
├── 11-gpu-acceleration/
├── 12-nvidia-libraries/
├── 13-gpu-image-filtering/
├── .gitignore
└── requirements.txt
```

## References

- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall.
- Oppenheim, A. V., Willsky, A. S., & Nawab, S. H. (1996). *Signals and Systems* (2nd ed.). Prentice Hall.

## Contributing

This is an educational project. Suggestions and improvements are welcome!

## License

MIT License
