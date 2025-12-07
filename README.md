# Digital Signal Processing: From Theory to GPU Acceleration

A comprehensive, hands-on tutorial covering digital signal processing from mathematical foundations through high-performance GPU implementations.

## Introduction

**Learn signal processing the right way** - starting with solid theoretical foundations, implementing algorithms from scratch, and scaling up to production-grade GPU acceleration.

This tutorial takes you on a complete journey through digital signal processing (DSP), combining rigorous theory from Oppenheim & Schafer's classic textbook with modern, practical implementations in Python. Unlike courses that treat theory and practice separately, each module here integrates mathematical understanding with working code, visualizations, and real-world applications.

### What Makes This Tutorial Unique?

- **Theory meets practice:** Every concept is both mathematically explained and implemented in Python
- **Ground-up progression:** From basic discrete signals to GPU-accelerated image filtering
- **Complete implementations:** No "magic boxes" - you'll build FFT, convolution, filters from scratch
- **Modern tools:** NumPy, SciPy, CuPy, Triton, and NVIDIA CUDA libraries
- **Real performance:** Achieve 10-200× speedups using GPU acceleration techniques
- **Production-ready:** Learn battle-tested NVIDIA libraries (cuFFT, NPP, Thrust)

### Who Is This For?

- **Students** learning DSP who want to see theory in action
- **Engineers** transitioning from MATLAB to Python
- **Developers** who need to implement high-performance signal/image processing
- **Researchers** requiring GPU acceleration for their algorithms
- **Self-learners** who want comprehensive coverage from basics to advanced

### What You'll Learn

**Fundamentals (Modules 1-4):**

- Discrete-time signals and systems
- Convolution and correlation
- DTFT, DFT, and FFT algorithms
- Frequency domain analysis

**Advanced Processing (Modules 5-10):**

- Filter design in time and frequency domains
- Windowing and spectral analysis
- 2D signal processing and image filtering
- Edge detection and morphological operations

**High-Performance Computing (Modules 11-13):**

- GPU programming with CUDA, CuPy, and Triton
- NVIDIA libraries: cuFFT (FFT), NPP (image processing), Thrust (algorithms)
- Optimized convolution and filtering on GPU
- **10-200× performance improvements** over CPU

### Prerequisites

- Basic Python programming
- Linear algebra (matrices, vectors)
- Calculus (derivatives, integrals)
- Understanding of complex numbers

No prior DSP knowledge required - we start from the beginning!

## Overview

This repository provides 13 comprehensive modules, each including theory documentation, Python implementations, visualizations, exercises, and practical examples. Progress sequentially for the best learning experience, or jump to specific topics using the navigation below.

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

```text
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
