"""
Module 13: GPU Image Filtering
Practical GPU-accelerated image processing implementations

Builds on Module 11 (GPU Fundamentals) and Module 12 (NVIDIA Libraries)
to demonstrate practical image filtering using CuPy, Triton, and custom CUDA.

Note: Requires CuPy and optionally Triton. Install with:
  pip install cupy-cuda11x  # for CUDA 11.x
  pip install cupy-cuda12x  # for CUDA 12.x
  pip install triton>=2.0.0 torch>=2.0.0  # for Triton examples

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

# Try to import Triton
try:
    import triton
    import triton.language as tl
    import torch
    TRITON_AVAILABLE = True
    print("Triton detected - Triton kernels will run")
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available - Triton examples will show conceptual code")


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
# Triton Kernels
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def conv2d_kernel(
        # Pointers to matrices
        image_ptr, kernel_ptr, output_ptr,
        # Image dimensions
        height, width,
        # Kernel dimensions
        ksize,
        # Strides
        stride_ih, stride_iw,
        stride_oh, stride_ow,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for 2D convolution"""
        # Get program ID
        pid = tl.program_id(axis=0)

        # Compute 2D position from 1D program ID
        num_blocks_w = tl.cdiv(width, BLOCK_SIZE)
        block_row = pid // num_blocks_w
        block_col = pid % num_blocks_w

        # Thread ID within block
        tid = tl.program_id(axis=1)

        # Compute output pixel position
        row = block_row * BLOCK_SIZE + (tid // BLOCK_SIZE)
        col = block_col * BLOCK_SIZE + (tid % BLOCK_SIZE)

        # Boundary check
        if row >= height or col >= width:
            return

        # Initialize accumulator
        acc = 0.0
        k_offset = ksize // 2

        # Convolve
        for ki in range(ksize):
            for kj in range(ksize):
                in_row = row + ki - k_offset
                in_col = col + kj - k_offset

                # Boundary check (zero padding)
                if in_row >= 0 and in_row < height and in_col >= 0 and in_col < width:
                    # Load image pixel
                    img_idx = in_row * stride_ih + in_col * stride_iw
                    img_val = tl.load(image_ptr + img_idx)

                    # Load kernel value
                    k_idx = ki * ksize + kj
                    k_val = tl.load(kernel_ptr + k_idx)

                    acc += img_val * k_val

        # Store result
        out_idx = row * stride_oh + col * stride_ow
        tl.store(output_ptr + out_idx, acc)


    @triton.jit
    def separable_row_kernel(
        input_ptr, output_ptr, kernel_ptr,
        height, width, ksize,
        stride_h, stride_w,
        BLOCK_SIZE: tl.constexpr,
    ):
        """First pass: convolve rows"""
        # Get position
        row = tl.program_id(axis=0)
        col_block = tl.program_id(axis=1)

        # Compute column range for this block
        col_start = col_block * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)

        # Mask for valid columns
        col_mask = cols < width

        k_offset = ksize // 2

        # Initialize accumulators for all columns in block
        accs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Convolve
        for k in range(ksize):
            # Compute source columns
            src_cols = cols + k - k_offset

            # Boundary mask
            valid_mask = col_mask & (src_cols >= 0) & (src_cols < width)

            # Load input values
            input_idx = row * stride_h + src_cols * stride_w
            input_vals = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)

            # Load kernel value
            k_val = tl.load(kernel_ptr + k)

            # Accumulate
            accs += input_vals * k_val

        # Store results
        output_idx = row * stride_h + cols * stride_w
        tl.store(output_ptr + output_idx, accs, mask=col_mask)


    @triton.jit
    def separable_col_kernel(
        input_ptr, output_ptr, kernel_ptr,
        height, width, ksize,
        stride_h, stride_w,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Second pass: convolve columns"""
        # Get position
        col = tl.program_id(axis=0)
        row_block = tl.program_id(axis=1)

        # Compute row range for this block
        row_start = row_block * BLOCK_SIZE
        rows = row_start + tl.arange(0, BLOCK_SIZE)

        # Mask for valid rows
        row_mask = rows < height

        k_offset = ksize // 2

        # Initialize accumulators
        accs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Convolve
        for k in range(ksize):
            # Compute source rows
            src_rows = rows + k - k_offset

            # Boundary mask
            valid_mask = row_mask & (src_rows >= 0) & (src_rows < height)

            # Load input values
            input_idx = src_rows * stride_h + col * stride_w
            input_vals = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)

            # Load kernel value
            k_val = tl.load(kernel_ptr + k)

            # Accumulate
            accs += input_vals * k_val

        # Store results
        output_idx = rows * stride_h + col * stride_w
        tl.store(output_ptr + output_idx, accs, mask=row_mask)


def triton_conv2d(image, kernel):
    """
    2D convolution using Triton

    Parameters:
    -----------
    image : torch.Tensor
        Input image (H, W)
    kernel : torch.Tensor
        Convolution kernel (K, K)

    Returns:
    --------
    output : torch.Tensor
        Filtered image
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    height, width = image.shape
    ksize = kernel.shape[0]

    # Allocate output
    output = torch.zeros_like(image)

    # Configure kernel launch
    BLOCK_SIZE = 16
    grid = lambda meta: (
        tl.cdiv(height, BLOCK_SIZE) * tl.cdiv(width, BLOCK_SIZE),
    )

    # Launch kernel
    conv2d_kernel[grid](
        image, kernel, output,
        height, width, ksize,
        image.stride(0), image.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def triton_separable_filter(image, kernel_1d):
    """
    Separable 2D filtering using Triton (two passes)

    Parameters:
    -----------
    image : torch.Tensor
        Input image
    kernel_1d : torch.Tensor
        1D kernel for separable filter

    Returns:
    --------
    output : torch.Tensor
        Filtered image
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    height, width = image.shape
    ksize = kernel_1d.shape[0]

    # Intermediate storage
    temp = torch.zeros_like(image)
    output = torch.zeros_like(image)

    BLOCK_SIZE = 16

    # Pass 1: Rows
    grid_rows = (height, tl.cdiv(width, BLOCK_SIZE))
    separable_row_kernel[grid_rows](
        image, temp, kernel_1d,
        height, width, ksize,
        image.stride(0), image.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2: Columns
    grid_cols = (width, tl.cdiv(height, BLOCK_SIZE))
    separable_col_kernel[grid_cols](
        temp, output, kernel_1d,
        height, width, ksize,
        temp.stride(0), temp.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def demonstrate_triton_filtering():
    """
    Demonstrate Triton-based image filtering
    """
    print("\nTriton Image Filtering Demo")
    print("=" * 50)

    if not TRITON_AVAILABLE:
        print("Triton not available - showing conceptual example")
        print("\nConceptual Triton 2D convolution:")
        print("""
        @triton.jit
        def conv2d_kernel(image_ptr, kernel_ptr, output_ptr,
                         height, width, ksize, ...):
            # Each program instance computes one output pixel
            row, col = get_position()

            acc = 0.0
            for ki in range(ksize):
                for kj in range(ksize):
                    acc += image[row+ki, col+kj] * kernel[ki, kj]

            output[row, col] = acc

        # Launch:
        grid = (num_blocks_h, num_blocks_w)
        conv2d_kernel[grid](image, kernel, output, ...)
        """)
        return

    # Create test image
    size = 512
    image_np = create_test_image(size, 'checkerboard')
    image_torch = torch.from_numpy(image_np).cuda()

    # Create kernel
    kernel_size = 11
    kernel_np = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    kernel_torch = torch.from_numpy(kernel_np).cuda()

    print(f"Image size: {size}×{size}")
    print(f"Kernel size: {kernel_size}×{kernel_size}")

    # Warm-up
    _ = triton_conv2d(image_torch, kernel_torch)
    torch.cuda.synchronize()

    # Timed run
    start = time.time()
    output = triton_conv2d(image_torch, kernel_torch)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) * 1000

    print(f"Triton 2D convolution time: {triton_time:.2f} ms")

    # Compare with CuPy if available
    if GPU_AVAILABLE:
        image_gpu = cp.asarray(image_np)
        kernel_gpu = cp.asarray(kernel_np)

        # Warm-up
        _ = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='constant')
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        output_cupy = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='constant')
        cp.cuda.Stream.null.synchronize()
        cupy_time = (time.time() - start) * 1000

        print(f"CuPy convolution time: {cupy_time:.2f} ms")
        print(f"Triton vs CuPy: {cupy_time / triton_time:.2f}x")

    # Test separable filter
    print("\nTriton Separable Filter (Gaussian-like)")
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32).cuda()
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Warm-up
    _ = triton_separable_filter(image_torch, kernel_1d)
    torch.cuda.synchronize()

    # Timed run
    start = time.time()
    output_sep = triton_separable_filter(image_torch, kernel_1d)
    torch.cuda.synchronize()
    triton_sep_time = (time.time() - start) * 1000

    print(f"Triton separable filter time: {triton_sep_time:.2f} ms")
    print(f"Speedup vs 2D: {triton_time / triton_sep_time:.2f}x")


def compare_gpu_frameworks():
    """
    Compare CuPy, Triton, and CPU performance on image filtering
    """
    print("\nGPU Framework Comparison")
    print("=" * 70)

    size = 1024
    kernel_size = 15

    image_np = create_test_image(size, 'circle')
    kernel_np = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    print(f"Image: {size}×{size}, Kernel: {kernel_size}×{kernel_size}")
    print(f"{'Framework':<15} {'Time (ms)':<15} {'Speedup vs CPU':<15}")
    print("-" * 70)

    # CPU baseline
    start = time.time()
    _ = cpu_ndimage.convolve(image_np, kernel_np, mode='constant')
    cpu_time = (time.time() - start) * 1000

    print(f"{'NumPy (CPU)':<15} {cpu_time:>10.2f}      {1.0:>10.1f}x")

    # CuPy
    if GPU_AVAILABLE:
        image_gpu = cp.asarray(image_np)
        kernel_gpu = cp.asarray(kernel_np)

        # Warm-up
        _ = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='constant')
        cp.cuda.Stream.null.synchronize()

        # Timed
        start = time.time()
        _ = gpu_ndimage.convolve(image_gpu, kernel_gpu, mode='constant')
        cp.cuda.Stream.null.synchronize()
        cupy_time = (time.time() - start) * 1000

        print(f"{'CuPy':<15} {cupy_time:>10.2f}      {cpu_time/cupy_time:>10.1f}x")
    else:
        print(f"{'CuPy':<15} {'N/A':>10}      {'N/A':>10}")

    # Triton
    if TRITON_AVAILABLE:
        image_torch = torch.from_numpy(image_np).cuda()
        kernel_torch = torch.from_numpy(kernel_np).cuda()

        # Warm-up
        _ = triton_conv2d(image_torch, kernel_torch)
        torch.cuda.synchronize()

        # Timed
        start = time.time()
        _ = triton_conv2d(image_torch, kernel_torch)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000

        print(f"{'Triton':<15} {triton_time:>10.2f}      {cpu_time/triton_time:>10.1f}x")
    else:
        print(f"{'Triton':<15} {'N/A':>10}      {'N/A':>10}")

    print("\nKey Observations:")
    print("- CuPy: Easiest to use, excellent performance for standard operations")
    print("- Triton: Good performance, more control, Python-based kernels")
    print("- Both show significant speedups over CPU for large images")


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

    # Example 8: Triton filtering
    print("\n" + "=" * 70)
    demonstrate_triton_filtering()

    # Example 9: Framework comparison
    print("\n" + "=" * 70)
    compare_gpu_frameworks()

    print("\n" + "=" * 70)
    print("All examples completed!")

    # Summary
    if GPU_AVAILABLE:
        print("\nSummary of CuPy Speedups:")
        print(f"  Gaussian Blur:    {results1.get('speedup', 'N/A'):.1f}x")
        print(f"  Convolution:      {results2.get('speedup', 'N/A'):.1f}x")
        print(f"  Sobel Edges:      {results3.get('speedup', 'N/A'):.1f}x")
        print(f"  FFT Filtering:    {results4.get('speedup', 'N/A'):.1f}x")
        print(f"  Batch Processing: {results5.get('speedup', 'N/A'):.1f}x")

    if TRITON_AVAILABLE:
        print("\nTriton framework also available for custom kernel development!")


if __name__ == "__main__":
    run_all_examples()
