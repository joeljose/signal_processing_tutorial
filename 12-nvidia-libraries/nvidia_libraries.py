"""
Module 12: NVIDIA Libraries for Signal and Image Processing
Practical examples using cuFFT, NPP (via OpenCV CUDA), and Thrust (via CuPy)

Note: Requires:
  - CuPy (provides cuFFT and Thrust interfaces)
  - OpenCV with CUDA support (provides NPP interface)

Installation:
  pip install cupy-cuda11x  # or cupy-cuda12x
  pip install opencv-contrib-python  # For GPU support check
"""

import numpy as np
import time

# Try to import CuPy (cuFFT, Thrust)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy detected - cuFFT and Thrust examples will run")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - showing conceptual examples only")

# Try to import OpenCV CUDA (NPP)
try:
    import cv2
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        OPENCV_CUDA_AVAILABLE = True
        print("OpenCV CUDA detected - NPP examples will run")
    else:
        OPENCV_CUDA_AVAILABLE = False
        print("OpenCV CUDA not available - NPP examples will show conceptual code")
except (ImportError, AttributeError):
    OPENCV_CUDA_AVAILABLE = False
    print("OpenCV CUDA not available - NPP examples will show conceptual code")


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_image(size=512):
    """Create a test image"""
    image = np.zeros((size, size), dtype=np.uint8)
    # Checkerboard pattern
    square_size = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                image[i*square_size:(i+1)*square_size,
                      j*square_size:(j+1)*square_size] = 255
    return image


def create_test_signal(n=1024):
    """Create a test signal with multiple frequency components"""
    t = np.linspace(0, 1, n)
    # Mix of frequencies
    signal = (np.sin(2 * np.pi * 10 * t) +
              0.5 * np.sin(2 * np.pi * 25 * t) +
              0.3 * np.sin(2 * np.pi * 50 * t))
    # Add noise
    signal += 0.1 * np.random.randn(n)
    return signal.astype(np.float32)


# ============================================================================
# cuFFT Examples
# ============================================================================

def demonstrate_cufft_1d():
    """
    Demonstrate cuFFT for 1D signal processing
    """
    print("\ncuFFT 1D FFT Demonstration")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Conceptual cuFFT usage via CuPy:
        signal_gpu = cp.asarray(signal)
        F_gpu = cp.fft.fft(signal_gpu)  # Uses cuFFT internally
        magnitude = cp.abs(F_gpu)
        """)
        return

    # Create test signal
    N = 1024 * 1024  # 1M points
    signal = create_test_signal(N)

    print(f"Signal length: {N:,} points")

    # CPU FFT
    start = time.time()
    F_cpu = np.fft.fft(signal)
    cpu_time = (time.time() - start) * 1000

    # GPU FFT (cuFFT)
    signal_gpu = cp.asarray(signal)

    # Warm-up
    _ = cp.fft.fft(signal_gpu)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    F_gpu = cp.fft.fft(signal_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy FFT) time: {cpu_time:.2f} ms")
    print(f"GPU (cuFFT) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    # Verify accuracy
    F_gpu_cpu = cp.asnumpy(F_gpu)
    error = np.max(np.abs(F_cpu - F_gpu_cpu))
    print(f"Max error: {error:.2e}")


def demonstrate_cufft_2d():
    """
    Demonstrate cuFFT for 2D image FFT
    """
    print("\ncuFFT 2D FFT Demonstration")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Conceptual 2D FFT using cuFFT:
        image_gpu = cp.asarray(image)
        F = cp.fft.fft2(image_gpu)  # cuFFT 2D transform
        magnitude = cp.abs(cp.fft.fftshift(F))
        """)
        return

    # Create test image
    size = 2048
    image = create_test_image(size).astype(np.float32)

    print(f"Image size: {size}×{size}")

    # CPU FFT
    start = time.time()
    F_cpu = np.fft.fft2(image)
    cpu_time = (time.time() - start) * 1000

    # GPU FFT (cuFFT)
    image_gpu = cp.asarray(image)

    # Warm-up
    _ = cp.fft.fft2(image_gpu)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    F_gpu = cp.fft.fft2(image_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy FFT2) time: {cpu_time:.2f} ms")
    print(f"GPU (cuFFT) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def demonstrate_fft_convolution():
    """
    Fast convolution using cuFFT
    """
    print("\nFast Convolution using cuFFT")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # FFT-based convolution:
        F_image = cp.fft.fft2(image)
        F_kernel = cp.fft.fft2(kernel_padded)
        F_result = F_image * F_kernel
        result = cp.real(cp.fft.ifft2(F_result))
        """)
        return

    # Create image and large kernel
    image = create_test_image(1024).astype(np.float32)
    kernel_size = 51
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    print(f"Image: {image.shape}, Kernel: {kernel.shape}")

    # Spatial convolution (CPU)
    from scipy import signal as scipy_signal
    start = time.time()
    result_spatial = scipy_signal.convolve2d(image, kernel, mode='same')
    spatial_time = (time.time() - start) * 1000

    # FFT convolution (GPU)
    image_gpu = cp.asarray(image)
    kernel_gpu = cp.asarray(kernel)

    # Pad kernel to image size
    kernel_padded = cp.zeros_like(image_gpu)
    kernel_padded[:kernel_size, :kernel_size] = kernel_gpu
    kernel_padded = cp.roll(kernel_padded,
                            (-kernel_size // 2, -kernel_size // 2),
                            axis=(0, 1))

    # Warm-up
    F_img = cp.fft.fft2(image_gpu)
    F_ker = cp.fft.fft2(kernel_padded)
    _ = cp.real(cp.fft.ifft2(F_img * F_ker))
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    F_image = cp.fft.fft2(image_gpu)
    F_kernel = cp.fft.fft2(kernel_padded)
    F_result = F_image * F_kernel
    result_fft = cp.real(cp.fft.ifft2(F_result))
    cp.cuda.Stream.null.synchronize()
    fft_time = (time.time() - start) * 1000

    print(f"Spatial convolution (CPU): {spatial_time:.2f} ms")
    print(f"FFT convolution (cuFFT): {fft_time:.2f} ms")
    print(f"Speedup: {spatial_time / fft_time:.1f}x")
    print("\nNote: FFT convolution is faster for large kernels (>15×15)")


# ============================================================================
# NPP Examples (via OpenCV CUDA)
# ============================================================================

def demonstrate_npp_gaussian():
    """
    Demonstrate NPP Gaussian filter via OpenCV CUDA
    """
    print("\nNPP Gaussian Filter Demonstration")
    print("=" * 70)

    if not OPENCV_CUDA_AVAILABLE:
        print("OpenCV CUDA not available - showing conceptual example")
        print("""
        # Conceptual NPP Gaussian filter:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gpu_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.5
        )
        gpu_result = gpu_filter.apply(gpu_img)
        result = gpu_result.download()
        """)
        return

    # Create test image
    image = create_test_image(2048)

    print(f"Image size: {image.shape}")

    # CPU Gaussian
    start = time.time()
    result_cpu = cv2.GaussianBlur(image, (5, 5), 1.5)
    cpu_time = (time.time() - start) * 1000

    # GPU Gaussian (NPP)
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    # Create filter
    gpu_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.5
    )

    # Warm-up
    _ = gpu_filter.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()

    # Timed run
    start = time.time()
    gpu_result = gpu_filter.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()
    gpu_time = (time.time() - start) * 1000

    result_gpu = gpu_result.download()

    print(f"CPU (OpenCV) time: {cpu_time:.2f} ms")
    print(f"GPU (NPP) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def demonstrate_npp_sobel():
    """
    Demonstrate NPP Sobel edge detection
    """
    print("\nNPP Sobel Edge Detection")
    print("=" * 70)

    if not OPENCV_CUDA_AVAILABLE:
        print("OpenCV CUDA not available - showing conceptual example")
        print("""
        # Conceptual NPP Sobel:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gpu_sobel = cv2.cuda.createSobelFilter(
            cv2.CV_8UC1, cv2.CV_16SC1, 1, 0, 3
        )
        gpu_edges = gpu_sobel.apply(gpu_img)
        """)
        return

    image = create_test_image(2048)

    # CPU Sobel
    start = time.time()
    sobelx_cpu = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    sobely_cpu = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
    magnitude_cpu = np.sqrt(sobelx_cpu.astype(np.float32)**2 +
                            sobely_cpu.astype(np.float32)**2)
    cpu_time = (time.time() - start) * 1000

    # GPU Sobel (NPP)
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    gpu_sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_16SC1, 1, 0, 3)
    gpu_sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_16SC1, 0, 1, 3)

    # Warm-up
    _ = gpu_sobel_x.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()

    # Timed run
    start = time.time()
    gpu_gx = gpu_sobel_x.apply(gpu_img)
    gpu_gy = gpu_sobel_y.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (OpenCV) time: {cpu_time:.2f} ms")
    print(f"GPU (NPP) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def demonstrate_npp_morphology():
    """
    Demonstrate NPP morphological operations
    """
    print("\nNPP Morphological Operations")
    print("=" * 70)

    if not OPENCV_CUDA_AVAILABLE:
        print("OpenCV CUDA not available - showing conceptual example")
        print("""
        # Conceptual NPP morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gpu_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_ERODE, cv2.CV_8UC1, kernel
        )
        gpu_eroded = gpu_filter.apply(gpu_img)
        """)
        return

    image = create_test_image(1024)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # CPU morphology
    start = time.time()
    eroded_cpu = cv2.erode(image, kernel)
    dilated_cpu = cv2.dilate(image, kernel)
    cpu_time = (time.time() - start) * 1000

    # GPU morphology (NPP)
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    gpu_erode = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel)
    gpu_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)

    # Warm-up
    _ = gpu_erode.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()

    # Timed run
    start = time.time()
    gpu_eroded = gpu_erode.apply(gpu_img)
    gpu_dilated = gpu_dilate.apply(gpu_img)
    cv2.cuda.Stream.waitForCompletion()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (OpenCV) time: {cpu_time:.2f} ms")
    print(f"GPU (NPP) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


# ============================================================================
# Thrust Examples (via CuPy)
# ============================================================================

def demonstrate_thrust_reduce():
    """
    Demonstrate Thrust reduction operations
    """
    print("\nThrust Reduction Operations")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Conceptual Thrust reductions:
        data_gpu = cp.asarray(data)
        total = cp.sum(data_gpu)      # Thrust reduce
        minimum = cp.min(data_gpu)    # Thrust reduce
        maximum = cp.max(data_gpu)    # Thrust reduce
        """)
        return

    # Create large array
    N = 10_000_000
    data = np.random.randn(N).astype(np.float32)

    print(f"Array size: {N:,} elements")

    # CPU reductions
    start = time.time()
    sum_cpu = np.sum(data)
    min_cpu = np.min(data)
    max_cpu = np.max(data)
    cpu_time = (time.time() - start) * 1000

    # GPU reductions (Thrust)
    data_gpu = cp.asarray(data)

    # Warm-up
    _ = cp.sum(data_gpu)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    sum_gpu = cp.sum(data_gpu)
    min_gpu = cp.min(data_gpu)
    max_gpu = cp.max(data_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy) time: {cpu_time:.2f} ms")
    print(f"GPU (Thrust) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    # Verify
    print(f"\nVerification:")
    print(f"  Sum: {float(sum_gpu):.2f} (CPU: {sum_cpu:.2f})")
    print(f"  Min: {float(min_gpu):.2f} (CPU: {min_cpu:.2f})")
    print(f"  Max: {float(max_gpu):.2f} (CPU: {max_cpu:.2f})")


def demonstrate_thrust_sort():
    """
    Demonstrate Thrust sorting
    """
    print("\nThrust Sort Demonstration")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Conceptual Thrust sort:
        data_gpu = cp.asarray(data)
        sorted_gpu = cp.sort(data_gpu)  # Thrust radix sort
        """)
        return

    # Create array
    N = 10_000_000
    data = np.random.randn(N).astype(np.float32)

    print(f"Array size: {N:,} elements")

    # CPU sort
    start = time.time()
    sorted_cpu = np.sort(data)
    cpu_time = (time.time() - start) * 1000

    # GPU sort (Thrust)
    data_gpu = cp.asarray(data)

    # Warm-up
    _ = cp.sort(data_gpu)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    sorted_gpu = cp.sort(data_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy) time: {cpu_time:.2f} ms")
    print(f"GPU (Thrust radix sort) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def demonstrate_thrust_scan():
    """
    Demonstrate Thrust prefix sum (scan)
    """
    print("\nThrust Prefix Sum (Scan)")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Conceptual Thrust scan:
        data_gpu = cp.asarray(data)
        cumsum_gpu = cp.cumsum(data_gpu)  # Thrust inclusive scan
        """)
        return

    # Create array
    N = 10_000_000
    data = np.random.randn(N).astype(np.float32)

    print(f"Array size: {N:,} elements")

    # CPU cumsum
    start = time.time()
    cumsum_cpu = np.cumsum(data)
    cpu_time = (time.time() - start) * 1000

    # GPU cumsum (Thrust scan)
    data_gpu = cp.asarray(data)

    # Warm-up
    _ = cp.cumsum(data_gpu)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    cumsum_gpu = cp.cumsum(data_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy) time: {cpu_time:.2f} ms")
    print(f"GPU (Thrust scan) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    print("\nApplication: Integral images for fast box filtering")


def demonstrate_integral_image():
    """
    Demonstrate integral image computation using Thrust scan
    """
    print("\nIntegral Image using Thrust")
    print("=" * 70)

    if not CUPY_AVAILABLE:
        print("CuPy not available - showing conceptual example")
        print("""
        # Integral image using Thrust scan:
        # Step 1: Cumsum along rows
        integral = cp.cumsum(image, axis=1)
        # Step 2: Cumsum along columns
        integral = cp.cumsum(integral, axis=0)
        # Now can do O(1) box filtering!
        """)
        return

    # Create image
    image = create_test_image(2048).astype(np.float32)

    print(f"Image size: {image.shape}")

    # CPU integral image
    start = time.time()
    integral_cpu = np.cumsum(np.cumsum(image, axis=1), axis=0)
    cpu_time = (time.time() - start) * 1000

    # GPU integral image (Thrust)
    image_gpu = cp.asarray(image)

    # Warm-up
    _ = cp.cumsum(cp.cumsum(image_gpu, axis=1), axis=0)
    cp.cuda.Stream.null.synchronize()

    # Timed run
    start = time.time()
    integral_gpu = cp.cumsum(image_gpu, axis=1)
    integral_gpu = cp.cumsum(integral_gpu, axis=0)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"CPU (NumPy) time: {cpu_time:.2f} ms")
    print(f"GPU (Thrust) time: {gpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    print("\nIntegral images enable O(1) box filtering regardless of box size!")


# ============================================================================
# Complete Pipeline Example
# ============================================================================

def complete_gpu_pipeline():
    """
    Complete image processing pipeline using multiple NVIDIA libraries
    """
    print("\nComplete GPU Pipeline")
    print("=" * 70)
    print("Using cuFFT + NPP + Thrust together")

    if not (CUPY_AVAILABLE and OPENCV_CUDA_AVAILABLE):
        print("\nRequires both CuPy and OpenCV CUDA")
        print("Pipeline would include:")
        print("  1. NPP Gaussian filter (denoise)")
        print("  2. NPP Sobel filter (edges)")
        print("  3. cuFFT for frequency analysis")
        print("  4. Thrust for statistics")
        return

    # Create image
    image = create_test_image(1024)

    # Step 1: Denoise with NPP
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.5)
    gpu_denoised = gpu_filter.apply(gpu_img)

    # Step 2: Edge detection with NPP
    gpu_sobel = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_16SC1, 1, 1, 3)
    gpu_edges = gpu_sobel.apply(gpu_denoised)

    # Step 3: Frequency analysis with cuFFT
    edges_cp = cp.asarray(gpu_edges.download()).astype(cp.float32)
    F = cp.fft.fft2(edges_cp)
    magnitude = cp.abs(cp.fft.fftshift(F))

    # Step 4: Statistics with Thrust
    mean_freq = float(cp.mean(magnitude))
    std_freq = float(cp.std(magnitude))
    max_freq = float(cp.max(magnitude))

    print(f"\nPipeline completed successfully!")
    print(f"Frequency domain statistics:")
    print(f"  Mean: {mean_freq:.2f}")
    print(f"  Std: {std_freq:.2f}")
    print(f"  Max: {max_freq:.2f}")
    print("\nAll operations performed on GPU - minimal CPU-GPU transfers!")


# ============================================================================
# Main Examples
# ============================================================================

def run_all_examples():
    """
    Run all NVIDIA library examples
    """
    print("Module 12: NVIDIA Libraries for Signal and Image Processing")
    print("=" * 70)

    if CUPY_AVAILABLE:
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU: {gpu_info['name'].decode()}")
        print(f"Compute Capability: {gpu_info['major']}.{gpu_info['minor']}")
        print(f"Memory: {gpu_info['totalGlobalMem'] / 1e9:.1f} GB")

    # cuFFT examples
    print("\n" + "=" * 70)
    print("cuFFT Examples")
    print("=" * 70)
    demonstrate_cufft_1d()
    demonstrate_cufft_2d()
    demonstrate_fft_convolution()

    # NPP examples
    print("\n" + "=" * 70)
    print("NPP (NVIDIA Performance Primitives) Examples")
    print("=" * 70)
    demonstrate_npp_gaussian()
    demonstrate_npp_sobel()
    demonstrate_npp_morphology()

    # Thrust examples
    print("\n" + "=" * 70)
    print("Thrust Examples")
    print("=" * 70)
    demonstrate_thrust_reduce()
    demonstrate_thrust_sort()
    demonstrate_thrust_scan()
    demonstrate_integral_image()

    # Complete pipeline
    print("\n" + "=" * 70)
    complete_gpu_pipeline()

    print("\n" + "=" * 70)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
