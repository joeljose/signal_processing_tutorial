"""
Module 12: GPU Image Filtering
Practical GPU-accelerated image processing implementations

Note: Requires CuPy (CUDA). Install with:
  pip install cupy-cuda11x  # for CUDA 11.x
  pip install cupy-cuda12x  # for CUDA 12.x

For systems without GPU, examples show CPU fallback.
"""

import numpy as np
import time
from scipy import ndimage as cpu_ndimage

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy import ndimage as gpu_ndimage
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU-only fallbacks")


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_image(size=512, pattern='checkerboard'):
    """Create test images"""
    image = np.zeros((size, size), dtype=np.float32)

    if pattern == 'checkerboard':
        square_size = size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    image[i*square_size:(i+1)*square_size,
                          j*square_size:(j+1)*square_size] = 1.0

    elif pattern == 'circle':
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] = 1.0

    elif pattern == 'gradient':
        for i in range(size):
            image[:, i] = i / (size - 1)

    return image


# ============================================================================
# GPU Gaussian Blur
# ============================================================================

def gaussian_blur_comparison(image, sigma=5.0):
    """
    Compare CPU vs GPU Gaussian blur

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Gaussian standard deviation

    Returns:
    --------
    results : dict
        Timing and output results
    """
    print("\nGaussian Blur Comparison")
    print("=" * 50)
    print(f"Image size: {image.shape}")
    print(f"Sigma: {sigma}")

    results = {}

    # CPU version
    start = time.time()
    blurred_cpu = cpu_ndimage.gaussian_filter(image, sigma=sigma)
    cpu_time = (time.time() - start) * 1000
    results['cpu_time'] = cpu_time
    results['cpu_output'] = blurred_cpu

    print(f"CPU time: {cpu_time:.2f} ms")

    if GPU_AVAILABLE:
        # GPU version
        image_gpu = cp.asarray(image)

        # Warm-up
        _ = gpu_ndimage.gaussian_filter(image_gpu, sigma=sigma)
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        blurred_gpu = gpu_ndimage.gaussian_filter(image_gpu, sigma=sigma)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        results['gpu_time'] = gpu_time
        results['gpu_output'] = cp.asnumpy(blurred_gpu)
        results['speedup'] = cpu_time / gpu_time

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

        # Verify accuracy
        error = np.max(np.abs(blurred_cpu - results['gpu_output']))
        print(f"Max difference: {error:.6f}")

    return results


# ============================================================================
# GPU Convolution
# ============================================================================

def convolution_comparison(image, kernel):
    """
    Compare CPU vs GPU 2D convolution

    Parameters:
    -----------
    image : ndarray
        Input image
    kernel : ndarray
        Convolution kernel

    Returns:
    --------
    results : dict
        Timing and output results
    """
    print("\n2D Convolution Comparison")
    print("=" * 50)
    print(f"Image size: {image.shape}")
    print(f"Kernel size: {kernel.shape}")

    results = {}

    # CPU version
    start = time.time()
    filtered_cpu = cpu_ndimage.convolve(image, kernel, mode='reflect')
    cpu_time = (time.time() - start) * 1000
    results['cpu_time'] = cpu_time
    results['cpu_output'] = filtered_cpu

    print(f"CPU time: {cpu_time:.2f} ms")

    if GPU_AVAILABLE:
        # GPU version
        image_gpu = cp.asarray(image)
        kernel_gpu = cp.asarray(kernel)

        # Warm-up
        _ = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='reflect')
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        filtered_gpu = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='reflect')
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        results['gpu_time'] = gpu_time
        results['gpu_output'] = cp.asnumpy(filtered_gpu)
        results['speedup'] = cpu_time / gpu_time

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    return results


# ============================================================================
# GPU Sobel Edge Detection
# ============================================================================

def sobel_edge_detection(image):
    """
    GPU-accelerated Sobel edge detection

    Parameters:
    -----------
    image : ndarray
        Input image

    Returns:
    --------
    results : dict
        Gradient magnitude and timing
    """
    print("\nSobel Edge Detection")
    print("=" * 50)

    # Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    results = {}

    # CPU version
    start = time.time()
    Gx_cpu = cpu_ndimage.convolve(image, sobel_x, mode='reflect')
    Gy_cpu = cpu_ndimage.convolve(image, sobel_y, mode='reflect')
    magnitude_cpu = np.sqrt(Gx_cpu**2 + Gy_cpu**2)
    cpu_time = (time.time() - start) * 1000

    results['cpu_time'] = cpu_time
    results['cpu_output'] = magnitude_cpu

    print(f"CPU time: {cpu_time:.2f} ms")

    if GPU_AVAILABLE:
        # GPU version
        image_gpu = cp.asarray(image)
        sobel_x_gpu = cp.asarray(sobel_x)
        sobel_y_gpu = cp.asarray(sobel_y)

        # Warm-up
        _ = gpu_ndimage.convolve(image_gpu, sobel_x_gpu, mode='reflect')
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        Gx_gpu = gpu_ndimage.convolve(image_gpu, sobel_x_gpu, mode='reflect')
        Gy_gpu = gpu_ndimage.convolve(image_gpu, sobel_y_gpu, mode='reflect')
        magnitude_gpu = cp.sqrt(Gx_gpu**2 + Gy_gpu**2)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        results['gpu_time'] = gpu_time
        results['gpu_output'] = cp.asnumpy(magnitude_gpu)
        results['speedup'] = cpu_time / gpu_time

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    return results


# ============================================================================
# GPU FFT-Based Filtering
# ============================================================================

def fft_filtering(image, filter_func):
    """
    FFT-based filtering on GPU

    Parameters:
    -----------
    image : ndarray
        Input image
    filter_func : callable
        Function that creates filter in frequency domain
        Takes (M, N) and returns filter H

    Returns:
    --------
    results : dict
        Filtered image and timing
    """
    print("\nFFT-Based Filtering")
    print("=" * 50)

    M, N = image.shape
    results = {}

    # CPU version
    start = time.time()
    F_cpu = np.fft.fft2(image)
    H_cpu = filter_func(M, N)
    G_cpu = F_cpu * H_cpu
    filtered_cpu = np.real(np.fft.ifft2(G_cpu))
    cpu_time = (time.time() - start) * 1000

    results['cpu_time'] = cpu_time
    results['cpu_output'] = filtered_cpu

    print(f"CPU time: {cpu_time:.2f} ms")

    if GPU_AVAILABLE:
        # GPU version
        image_gpu = cp.asarray(image)
        H_gpu = cp.asarray(filter_func(M, N))

        # Warm-up
        _ = cp.fft.fft2(image_gpu)
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        F_gpu = cp.fft.fft2(image_gpu)
        G_gpu = F_gpu * H_gpu
        filtered_gpu = cp.real(cp.fft.ifft2(G_gpu))
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        results['gpu_time'] = gpu_time
        results['gpu_output'] = cp.asnumpy(filtered_gpu)
        results['speedup'] = cpu_time / gpu_time

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    return results


def create_lowpass_filter(M, N, cutoff=0.1):
    """Create ideal lowpass filter"""
    k = np.arange(M) - M // 2
    l = np.arange(N) - N // 2
    kk, ll = np.meshgrid(l, k)
    D = np.sqrt(kk**2 + ll**2)

    H = np.zeros((M, N))
    H[D <= cutoff * min(M, N)] = 1.0
    return np.fft.ifftshift(H)


# ============================================================================
# Batch Processing
# ============================================================================

def batch_processing_demo(num_images=10, size=512):
    """
    Demonstrate batch processing multiple images

    Parameters:
    -----------
    num_images : int
        Number of images to process
    size : int
        Image dimension

    Returns:
    --------
    results : dict
        Timing results
    """
    print("\nBatch Processing Demo")
    print("=" * 50)
    print(f"Processing {num_images} images of size {size}×{size}")

    # Create batch of images
    images = np.array([create_test_image(size, 'checkerboard')
                       for _ in range(num_images)], dtype=np.float32)

    results = {}

    # CPU: Process sequentially
    start = time.time()
    filtered_cpu = np.array([cpu_ndimage.gaussian_filter(img, sigma=3.0)
                             for img in images])
    cpu_time = (time.time() - start) * 1000

    results['cpu_time'] = cpu_time
    print(f"CPU time (sequential): {cpu_time:.2f} ms")
    print(f"  Time per image: {cpu_time / num_images:.2f} ms")

    if GPU_AVAILABLE:
        # GPU: Batch process
        images_gpu = cp.asarray(images)

        # Warm-up
        for img_gpu in images_gpu:
            _ = gpu_ndimage.gaussian_filter(img_gpu, sigma=3.0)
        cp.cuda.Stream.null.synchronize()

        # Timed batch processing
        start = time.time()
        filtered_gpu = cp.array([gpu_ndimage.gaussian_filter(img, sigma=3.0)
                                 for img in images_gpu])
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time

        print(f"GPU time (batch): {gpu_time:.2f} ms")
        print(f"  Time per image: {gpu_time / num_images:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    return results


# ============================================================================
# Performance Scaling Analysis
# ============================================================================

def analyze_scaling():
    """
    Analyze how performance scales with image size
    """
    print("\nPerformance Scaling Analysis")
    print("=" * 50)

    sizes = [128, 256, 512, 1024, 2048]
    sigma = 5.0

    print(f"\nGaussian Blur (σ={sigma})")
    print(f"{'Size':<12} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for size in sizes:
        image = create_test_image(size, 'circle')

        # CPU
        start = time.time()
        _ = cpu_ndimage.gaussian_filter(image, sigma=sigma)
        cpu_time = (time.time() - start) * 1000

        if GPU_AVAILABLE:
            # GPU
            image_gpu = cp.asarray(image)

            # Warm-up
            _ = gpu_ndimage.gaussian_filter(image_gpu, sigma=sigma)
            cp.cuda.Stream.null.synchronize()

            # Timed
            start = time.time()
            _ = gpu_ndimage.gaussian_filter(image_gpu, sigma=sigma)
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - start) * 1000

            speedup = cpu_time / gpu_time

            print(f"{size}×{size:<6} {cpu_time:>10.2f}      {gpu_time:>10.2f}      {speedup:>6.1f}x")
        else:
            print(f"{size}×{size:<6} {cpu_time:>10.2f}      N/A             N/A")

    if GPU_AVAILABLE:
        print("\nKey observation: Speedup increases with image size!")
        print("Larger images amortize GPU overhead better.")


# ============================================================================
# Custom Kernel Example (Conceptual)
# ============================================================================

def demonstrate_custom_kernel():
    """
    Show custom CUDA kernel implementation (conceptual)
    """
    print("\nCustom CUDA Kernel Example (Conceptual)")
    print("=" * 50)

    if not GPU_AVAILABLE:
        print("Requires CuPy/CUDA")
        print("\nConceptual custom kernel for brightness adjustment:")
        print("""
        kernel = cp.RawKernel(r'''
        extern "C" __global__
        void brighten(const float* input, float* output, float factor, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < size) {
                output[idx] = input[idx] * factor;
            }
        }
        ''', 'brighten')

        # Launch:
        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,),
               (input_gpu, output_gpu, factor, size))
        """)
        return

    # Simple brightness kernel
    brighten_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void brighten(const float* input, float* output, float factor, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            output[idx] = input[idx] * factor;
        }
    }
    ''', 'brighten')

    # Test
    size = 1024 * 1024
    image_flat = cp.random.rand(size, dtype=cp.float32)
    output = cp.zeros_like(image_flat)
    factor = 1.5

    # Configure kernel launch
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    # Launch kernel
    start = time.time()
    brighten_kernel((blocks,), (threads_per_block,),
                    (image_flat, output, factor, size))
    cp.cuda.Stream.null.synchronize()
    kernel_time = (time.time() - start) * 1000

    # Compare with built-in
    start = time.time()
    output_builtin = image_flat * factor
    cp.cuda.Stream.null.synchronize()
    builtin_time = (time.time() - start) * 1000

    print(f"Custom kernel time: {kernel_time:.3f} ms")
    print(f"Built-in operation time: {builtin_time:.3f} ms")
    print("Note: Built-in operations are highly optimized!")


# ============================================================================
# Examples
# ============================================================================

def run_all_examples():
    """
    Run all GPU image filtering examples
    """
    print("Module 12: GPU Image Filtering")
    print("=" * 70)

    if GPU_AVAILABLE:
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU: {gpu_info['name'].decode()}")
        print(f"Compute Capability: {gpu_info['major']}.{gpu_info['minor']}")
        print(f"Memory: {gpu_info['totalGlobalMem'] / 1e9:.1f} GB")
    else:
        print("\nNo GPU detected - showing CPU-only results")

    # Create test image
    image = create_test_image(512, 'checkerboard')

    # Example 1: Gaussian blur
    print("\n" + "=" * 70)
    results1 = gaussian_blur_comparison(image, sigma=5.0)

    # Example 2: Convolution
    print("\n" + "=" * 70)
    kernel = np.ones((21, 21), dtype=np.float32) / (21 * 21)
    results2 = convolution_comparison(image, kernel)

    # Example 3: Sobel edge detection
    print("\n" + "=" * 70)
    results3 = sobel_edge_detection(image)

    # Example 4: FFT filtering
    print("\n" + "=" * 70)
    results4 = fft_filtering(image, create_lowpass_filter)

    # Example 5: Batch processing
    print("\n" + "=" * 70)
    results5 = batch_processing_demo(num_images=10, size=512)

    # Example 6: Scaling analysis
    print("\n" + "=" * 70)
    analyze_scaling()

    # Example 7: Custom kernel
    print("\n" + "=" * 70)
    demonstrate_custom_kernel()

    print("\n" + "=" * 70)
    print("All examples completed!")

    # Summary
    if GPU_AVAILABLE:
        print("\nSummary of Speedups:")
        print(f"  Gaussian Blur:    {results1.get('speedup', 'N/A'):.1f}x")
        print(f"  Convolution:      {results2.get('speedup', 'N/A'):.1f}x")
        print(f"  Sobel Edges:      {results3.get('speedup', 'N/A'):.1f}x")
        print(f"  FFT Filtering:    {results4.get('speedup', 'N/A'):.1f}x")
        print(f"  Batch Processing: {results5.get('speedup', 'N/A'):.1f}x")


if __name__ == "__main__":
    run_all_examples()
