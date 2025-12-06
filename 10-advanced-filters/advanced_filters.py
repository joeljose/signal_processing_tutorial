"""
Module 10: Advanced Image Filters
Comprehensive implementation of advanced image filtering techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
import time

# ============================================================================
# Gaussian Filtering
# ============================================================================

def create_gaussian_kernel(size, sigma):
    """
    Create 2D Gaussian kernel

    Parameters:
    -----------
    size : int
        Kernel size (should be odd)
    sigma : float
        Standard deviation

    Returns:
    --------
    kernel : ndarray
        Normalized Gaussian kernel
    """
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def separable_gaussian_filter(image, sigma):
    """
    Apply Gaussian filter using separable implementation

    Much faster than 2D convolution!

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Standard deviation

    Returns:
    --------
    filtered : ndarray
        Filtered image
    """
    # Kernel size: 6*sigma captures 99.7% of energy
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    # Create 1D Gaussian kernel
    center = size // 2
    x = np.arange(size) - center
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= np.sum(kernel_1d)

    # Apply separably
    filtered = ndimage.convolve1d(image, kernel_1d, axis=0, mode='reflect')
    filtered = ndimage.convolve1d(filtered, kernel_1d, axis=1, mode='reflect')

    return filtered


def gaussian_pyramid(image, scales, sigma_base=1.0, k=np.sqrt(2)):
    """
    Create Gaussian pyramid at multiple scales

    Parameters:
    -----------
    image : ndarray
        Input image
    scales : int
        Number of scales
    sigma_base : float
        Base sigma
    k : float
        Scale factor between levels

    Returns:
    --------
    pyramid : list of ndarray
        List of smoothed images at different scales
    sigmas : list of float
        Sigma values used
    """
    pyramid = [image.copy()]
    sigmas = [0]

    for s in range(1, scales):
        sigma = sigma_base * (k ** s)
        smoothed = separable_gaussian_filter(image, sigma)
        pyramid.append(smoothed)
        sigmas.append(sigma)

    return pyramid, sigmas


# ============================================================================
# Gradient-Based Edge Detection
# ============================================================================

def sobel_filters():
    """
    Sobel edge detection kernels

    Returns:
    --------
    sobel_x, sobel_y : ndarray
        Horizontal and vertical Sobel kernels
    """
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

    return sobel_x, sobel_y


def prewitt_filters():
    """
    Prewitt edge detection kernels

    Returns:
    --------
    prewitt_x, prewitt_y : ndarray
        Horizontal and vertical Prewitt kernels
    """
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    return prewitt_x, prewitt_y


def roberts_filters():
    """
    Roberts cross edge detection kernels

    Returns:
    --------
    roberts_x, roberts_y : ndarray
        Diagonal gradient kernels
    """
    roberts_x = np.array([
        [ 1, 0],
        [ 0, -1]
    ], dtype=np.float32)

    roberts_y = np.array([
        [0,  1],
        [-1, 0]
    ], dtype=np.float32)

    return roberts_x, roberts_y


def scharr_filters():
    """
    Scharr edge detection kernels (improved Sobel)

    Better rotational symmetry than Sobel

    Returns:
    --------
    scharr_x, scharr_y : ndarray
        Horizontal and vertical Scharr kernels
    """
    scharr_x = np.array([
        [ -3, 0,  3],
        [-10, 0, 10],
        [ -3, 0,  3]
    ], dtype=np.float32)

    scharr_y = np.array([
        [-3, -10, -3],
        [ 0,   0,  0],
        [ 3,  10,  3]
    ], dtype=np.float32)

    return scharr_x, scharr_y


def compute_gradients(image, method='sobel'):
    """
    Compute image gradients

    Parameters:
    -----------
    image : ndarray
        Input image
    method : str
        'sobel', 'prewitt', 'roberts', or 'scharr'

    Returns:
    --------
    Gx, Gy : ndarray
        Gradient in x and y directions
    magnitude : ndarray
        Gradient magnitude
    direction : ndarray
        Gradient direction in radians
    """
    # Select filter
    if method == 'sobel':
        kernel_x, kernel_y = sobel_filters()
    elif method == 'prewitt':
        kernel_x, kernel_y = prewitt_filters()
    elif method == 'roberts':
        kernel_x, kernel_y = roberts_filters()
    elif method == 'scharr':
        kernel_x, kernel_y = scharr_filters()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute gradients
    Gx = ndimage.convolve(image, kernel_x, mode='reflect')
    Gy = ndimage.convolve(image, kernel_y, mode='reflect')

    # Magnitude and direction
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)

    return Gx, Gy, magnitude, direction


# ============================================================================
# Second-Order Derivatives
# ============================================================================

def laplacian_kernel(connectivity=4):
    """
    Laplacian kernel

    Parameters:
    -----------
    connectivity : int
        4 or 8 (4-connected or 8-connected)

    Returns:
    --------
    kernel : ndarray
        Laplacian kernel
    """
    if connectivity == 4:
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
    elif connectivity == 8:
        kernel = np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=np.float32)
    else:
        raise ValueError("Connectivity must be 4 or 8")

    return kernel


def apply_laplacian(image, connectivity=4):
    """
    Apply Laplacian operator

    Parameters:
    -----------
    image : ndarray
        Input image
    connectivity : int
        4 or 8

    Returns:
    --------
    laplacian : ndarray
        Laplacian-filtered image
    """
    kernel = laplacian_kernel(connectivity)
    return ndimage.convolve(image, kernel, mode='reflect')


def laplacian_of_gaussian(size, sigma):
    """
    Create Laplacian of Gaussian (LoG) kernel

    Parameters:
    -----------
    size : int
        Kernel size (should be odd)
    sigma : float
        Standard deviation

    Returns:
    --------
    kernel : ndarray
        LoG kernel
    """
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]

    # LoG formula
    kernel = -(1 / (np.pi * sigma**4)) * \
             (1 - (x**2 + y**2) / (2 * sigma**2)) * \
             np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize to sum to zero (DC component = 0)
    kernel = kernel - np.mean(kernel)

    return kernel


def apply_log(image, sigma):
    """
    Apply Laplacian of Gaussian filter

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Standard deviation

    Returns:
    --------
    log_image : ndarray
        LoG-filtered image
    """
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    kernel = laplacian_of_gaussian(size, sigma)
    return ndimage.convolve(image, kernel, mode='reflect')


def difference_of_gaussians(image, sigma1, sigma2=None):
    """
    Compute Difference of Gaussians (DoG)

    Approximates LoG, used in SIFT

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma1 : float
        Smaller sigma
    sigma2 : float, optional
        Larger sigma (default: 1.6 * sigma1)

    Returns:
    --------
    dog : ndarray
        DoG-filtered image
    """
    if sigma2 is None:
        sigma2 = 1.6 * sigma1

    # Ensure sigma2 > sigma1
    if sigma2 < sigma1:
        sigma1, sigma2 = sigma2, sigma1

    # Compute two Gaussian-smoothed images
    smooth1 = separable_gaussian_filter(image, sigma1)
    smooth2 = separable_gaussian_filter(image, sigma2)

    # Difference
    dog = smooth2 - smooth1

    return dog


# ============================================================================
# Canny Edge Detector
# ============================================================================

def canny_edge_detector(image, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
    """
    Full Canny edge detection algorithm

    Steps:
    1. Gaussian smoothing
    2. Gradient computation
    3. Non-maximum suppression
    4. Double thresholding
    5. Edge tracking by hysteresis

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Gaussian smoothing parameter
    low_threshold : float
        Low threshold for hysteresis (fraction of max)
    high_threshold : float
        High threshold for hysteresis (fraction of max)

    Returns:
    --------
    edges : ndarray
        Binary edge map
    """
    # Step 1: Gaussian smoothing
    smoothed = separable_gaussian_filter(image, sigma)

    # Step 2: Compute gradients
    Gx, Gy, magnitude, direction = compute_gradients(smoothed, method='sobel')

    # Step 3: Non-maximum suppression
    nms = non_maximum_suppression(magnitude, direction)

    # Step 4: Double thresholding
    max_val = np.max(nms)
    T_low = low_threshold * max_val
    T_high = high_threshold * max_val

    strong_edges = nms >= T_high
    weak_edges = (nms >= T_low) & (nms < T_high)

    # Step 5: Edge tracking by hysteresis
    edges = edge_tracking_hysteresis(strong_edges, weak_edges)

    return edges


def non_maximum_suppression(magnitude, direction):
    """
    Suppress non-maximum gradients along gradient direction

    Thins edges to single-pixel width

    Parameters:
    -----------
    magnitude : ndarray
        Gradient magnitude
    direction : ndarray
        Gradient direction in radians

    Returns:
    --------
    suppressed : ndarray
        Magnitude after NMS
    """
    M, N = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Convert direction to degrees [0, 180)
    angle = direction * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Determine neighbors along gradient direction
            q = 255
            r = 255

            # Angle 0° (horizontal edge, check vertical neighbors)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45° (diagonal)
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j + 1]
                r = magnitude[i - 1, j - 1]
            # Angle 90° (vertical edge, check horizontal neighbors)
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135° (diagonal)
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            # Keep only if local maximum
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def edge_tracking_hysteresis(strong_edges, weak_edges):
    """
    Track edges using hysteresis

    Keep weak edges only if connected to strong edges

    Parameters:
    -----------
    strong_edges : ndarray (boolean)
        Strong edge pixels
    weak_edges : ndarray (boolean)
        Weak edge pixels

    Returns:
    --------
    edges : ndarray (boolean)
        Final edge map
    """
    M, N = strong_edges.shape
    edges = strong_edges.copy()

    # Iteratively add weak edges connected to strong edges
    changed = True
    while changed:
        changed = False

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if weak_edges[i, j] and not edges[i, j]:
                    # Check if any 8-connected neighbor is a strong edge
                    if np.any(edges[i-1:i+2, j-1:j+2]):
                        edges[i, j] = True
                        changed = True

    return edges


# ============================================================================
# Image Sharpening
# ============================================================================

def unsharp_masking(image, sigma=1.0, amount=1.0):
    """
    Unsharp masking for image sharpening

    Algorithm:
    1. Blur image
    2. Subtract blur from original (high-pass)
    3. Add back scaled high-pass to original

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Blur amount
    amount : float
        Sharpening strength (α)

    Returns:
    --------
    sharpened : ndarray
        Sharpened image
    """
    # Blur
    blurred = separable_gaussian_filter(image, sigma)

    # Mask (high-frequency component)
    mask = image - blurred

    # Sharpen
    sharpened = image + amount * mask

    return sharpened


def high_boost_filter(image, sigma=1.0, A=1.5):
    """
    High-boost filtering

    I_boost = A * I - G_σ * I

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Blur amount
    A : float
        Boost factor (A ≥ 1)

    Returns:
    --------
    boosted : ndarray
        High-boost filtered image
    """
    blurred = separable_gaussian_filter(image, sigma)
    boosted = A * image - blurred

    return boosted


# ============================================================================
# Edge-Preserving Filters
# ============================================================================

def bilateral_filter(image, sigma_spatial=3.0, sigma_range=0.1, kernel_size=None):
    """
    Bilateral filter for edge-preserving smoothing

    Combines spatial and range (intensity) Gaussians

    Parameters:
    -----------
    image : ndarray
        Input image (values in [0, 1] recommended)
    sigma_spatial : float
        Spatial standard deviation
    sigma_range : float
        Range (intensity) standard deviation
    kernel_size : int, optional
        Kernel size (default: 6*sigma_spatial + 1)

    Returns:
    --------
    filtered : ndarray
        Filtered image
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma_spatial + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    M, N = image.shape
    filtered = np.zeros_like(image)

    # Half kernel size
    offset = kernel_size // 2

    # Create spatial Gaussian kernel
    x = np.arange(kernel_size) - offset
    xx, yy = np.meshgrid(x, x)
    spatial_gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma_spatial**2))

    # Process each pixel
    for i in range(M):
        for j in range(N):
            # Extract neighborhood
            i_min = max(i - offset, 0)
            i_max = min(i + offset + 1, M)
            j_min = max(j - offset, 0)
            j_max = min(j + offset + 1, N)

            neighborhood = image[i_min:i_max, j_min:j_max]

            # Corresponding spatial weights
            spatial_window = spatial_gaussian[
                (i_min - (i - offset)):(i_max - (i - offset)),
                (j_min - (j - offset)):(j_max - (j - offset))
            ]

            # Range (intensity) weights
            intensity_diff = neighborhood - image[i, j]
            range_gaussian = np.exp(-intensity_diff**2 / (2 * sigma_range**2))

            # Combined weight
            weight = spatial_window * range_gaussian
            weight_sum = np.sum(weight)

            # Weighted average
            if weight_sum > 0:
                filtered[i, j] = np.sum(weight * neighborhood) / weight_sum
            else:
                filtered[i, j] = image[i, j]

    return filtered


# ============================================================================
# Test Image Creation
# ============================================================================

def create_test_image(size=256, pattern='checkerboard'):
    """Create test images"""
    image = np.zeros((size, size))

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
# Examples
# ============================================================================

def example1_gradient_operators():
    """
    Example 1: Compare gradient operators
    """
    print("Example 1: Gradient Operator Comparison")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'circle')

    # Add noise
    image_noisy = image + 0.1 * np.random.randn(size, size)
    image_noisy = np.clip(image_noisy, 0, 1)

    methods = ['sobel', 'prewitt', 'scharr', 'roberts']
    results = []

    for method in methods:
        _, _, magnitude, _ = compute_gradients(image_noisy, method=method)
        results.append(magnitude)
        print(f"{method.capitalize():10s}: Max gradient = {np.max(magnitude):.3f}")

    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(image_noisy, cmap='gray')
    axes[0].set_title('Original (Noisy)')
    axes[0].axis('off')

    for i, (mag, method) in enumerate(zip(results, methods)):
        axes[i+1].imshow(mag, cmap='gray')
        axes[i+1].set_title(f'{method.capitalize()} Gradient')
        axes[i+1].axis('off')

    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


def example2_laplacian_log():
    """
    Example 2: Laplacian vs LoG vs DoG
    """
    print("\nExample 2: Second-Order Derivatives")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'circle')

    # Apply filters
    laplacian_4 = apply_laplacian(image, connectivity=4)
    laplacian_8 = apply_laplacian(image, connectivity=8)
    log = apply_log(image, sigma=2.0)
    dog = difference_of_gaussians(image, sigma1=1.0, sigma2=1.6)

    # Display
    images = [image, laplacian_4, laplacian_8, log, dog]
    titles = ['Original', 'Laplacian (4-conn)', 'Laplacian (8-conn)',
              'LoG (σ=2)', 'DoG (σ1=1, σ2=1.6)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

    print("Laplacian: Very noise-sensitive")
    print("LoG: Smoothing reduces noise sensitivity")
    print("DoG: Efficient approximation to LoG")


def example3_canny_detector():
    """
    Example 3: Canny edge detector with different parameters
    """
    print("\nExample 3: Canny Edge Detector")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'checkerboard')

    # Add noise
    image_noisy = image + 0.05 * np.random.randn(size, size)
    image_noisy = np.clip(image_noisy, 0, 1)

    # Test different parameters
    params = [
        (1.0, 0.05, 0.15),  # sigma, low_threshold, high_threshold
        (1.0, 0.10, 0.30),
        (2.0, 0.10, 0.30),
    ]

    results = []
    for sigma, low, high in params:
        edges = canny_edge_detector(image_noisy, sigma=sigma,
                                    low_threshold=low, high_threshold=high)
        results.append(edges)
        print(f"σ={sigma}, T_low={low}, T_high={high}")

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].imshow(image_noisy, cmap='gray')
    axes[0].set_title('Original (Noisy)')
    axes[0].axis('off')

    for i, (edges, (sigma, low, high)) in enumerate(zip(results, params)):
        axes[i+1].imshow(edges, cmap='gray')
        axes[i+1].set_title(f'Canny (σ={sigma}, T_low={low}, T_high={high})')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()


def example4_unsharp_masking():
    """
    Example 4: Unsharp masking
    """
    print("\nExample 4: Unsharp Masking")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'checkerboard')

    # Blur image slightly
    blurred = separable_gaussian_filter(image, sigma=2.0)

    # Sharpen with different amounts
    amounts = [0.5, 1.0, 2.0]
    results = []

    for amount in amounts:
        sharpened = unsharp_masking(blurred, sigma=1.0, amount=amount)
        results.append(sharpened)
        print(f"Sharpening amount α = {amount}")

    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(blurred, cmap='gray')
    axes[0].set_title('Blurred Original')
    axes[0].axis('off')

    for i, (sharp, amount) in enumerate(zip(results, amounts)):
        axes[i+1].imshow(sharp, cmap='gray')
        axes[i+1].set_title(f'Unsharp Mask (α={amount})')
        axes[i+1].axis('off')

    axes[4].axis('off')
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


def example5_bilateral_filter():
    """
    Example 5: Bilateral filtering (edge-preserving)
    """
    print("\nExample 5: Bilateral Filter (Edge-Preserving)")
    print("=" * 50)

    size = 128  # Smaller for computational efficiency
    image = create_test_image(size, 'circle')

    # Add noise
    image_noisy = image + 0.2 * np.random.randn(size, size)
    image_noisy = np.clip(image_noisy, 0, 1)

    # Compare Gaussian vs Bilateral
    print("Filtering (this may take a moment)...")

    gaussian_filtered = separable_gaussian_filter(image_noisy, sigma=2.0)

    bilateral_filtered = bilateral_filter(image_noisy,
                                         sigma_spatial=2.0,
                                         sigma_range=0.1,
                                         kernel_size=11)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_noisy, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[0].axis('off')

    axes[1].imshow(gaussian_filtered, cmap='gray')
    axes[1].set_title('Gaussian Filter (Blurs Edges)')
    axes[1].axis('off')

    axes[2].imshow(bilateral_filtered, cmap='gray')
    axes[2].set_title('Bilateral Filter (Preserves Edges)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Gaussian: Smooths everything, including edges")
    print("Bilateral: Smooths flat regions, preserves edges")


def example6_gaussian_pyramid():
    """
    Example 6: Gaussian pyramid (multi-scale representation)
    """
    print("\nExample 6: Gaussian Pyramid")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'checkerboard')

    # Create pyramid
    pyramid, sigmas = gaussian_pyramid(image, scales=5, sigma_base=1.0, k=np.sqrt(2))

    print(f"Created pyramid with {len(pyramid)} scales")
    for i, sigma in enumerate(sigmas):
        print(f"Level {i}: σ = {sigma:.2f}")

    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(6, len(pyramid))):
        axes[i].imshow(pyramid[i], cmap='gray')
        axes[i].set_title(f'Scale {i} (σ={sigmas[i]:.2f})')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def example7_filter_timing():
    """
    Example 7: Filter timing comparisons
    """
    print("\nExample 7: Filter Performance Comparison")
    print("=" * 50)

    sizes = [128, 256, 512]

    print(f"\n{'Filter':<20} {'128×128 (ms)':<15} {'256×256 (ms)':<15} {'512×512 (ms)':<15}")
    print("-" * 70)

    for size in sizes:
        image = create_test_image(size, 'checkerboard')

        # Gaussian (separable)
        if size == 128:
            start = time.time()
            _ = separable_gaussian_filter(image, sigma=2.0)
            time_gaussian = (time.time() - start) * 1000
            times_gaussian = [time_gaussian]
        elif size == 256:
            start = time.time()
            _ = separable_gaussian_filter(image, sigma=2.0)
            times_gaussian.append((time.time() - start) * 1000)
        else:
            start = time.time()
            _ = separable_gaussian_filter(image, sigma=2.0)
            times_gaussian.append((time.time() - start) * 1000)

        # Sobel
        if size == 128:
            start = time.time()
            _ = compute_gradients(image, method='sobel')
            time_sobel = (time.time() - start) * 1000
            times_sobel = [time_sobel]
        elif size == 256:
            start = time.time()
            _ = compute_gradients(image, method='sobel')
            times_sobel.append((time.time() - start) * 1000)
        else:
            start = time.time()
            _ = compute_gradients(image, method='sobel')
            times_sobel.append((time.time() - start) * 1000)

    print(f"{'Gaussian (separable)':<20} {times_gaussian[0]:>10.2f}      {times_gaussian[1]:>10.2f}      {times_gaussian[2]:>10.2f}")
    print(f"{'Sobel gradient':<20} {times_sobel[0]:>10.2f}      {times_sobel[1]:>10.2f}      {times_sobel[2]:>10.2f}")

    print("\nNote: Separable filters scale better with image size!")


def run_all_examples():
    """
    Run all examples
    """
    print("Module 10: Advanced Image Filters")
    print("=" * 70)

    example1_gradient_operators()
    example2_laplacian_log()
    example3_canny_detector()
    example4_unsharp_masking()
    example5_bilateral_filter()
    example6_gaussian_pyramid()
    example7_filter_timing()

    print("\n" + "=" * 70)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
