"""
Module 8: 2D Signals and Images
Comprehensive implementation of 2D signal processing and image fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal, ndimage
from PIL import Image
import time

# ============================================================================
# Image Loading and Representation
# ============================================================================

def load_image(filepath, mode='grayscale'):
    """
    Load image from file

    Parameters:
    -----------
    filepath : str
        Path to image file
    mode : str
        'grayscale', 'rgb', or 'original'

    Returns:
    --------
    image : ndarray
        Image array, normalized to [0, 1]
    """
    img = Image.open(filepath)

    if mode == 'grayscale':
        img = img.convert('L')  # Convert to grayscale
        image = np.array(img, dtype=np.float32) / 255.0
    elif mode == 'rgb':
        img = img.convert('RGB')
        image = np.array(img, dtype=np.float32) / 255.0
    else:
        image = np.array(img, dtype=np.float32) / 255.0

    return image


def create_test_image(size=256, pattern='checkerboard'):
    """
    Create synthetic test images

    Parameters:
    -----------
    size : int
        Image dimension (size x size)
    pattern : str
        'checkerboard', 'gradient', 'circle', 'bars'

    Returns:
    --------
    image : ndarray
        Test image of shape (size, size)
    """
    image = np.zeros((size, size))

    if pattern == 'checkerboard':
        # Checkerboard pattern
        square_size = size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    image[i*square_size:(i+1)*square_size,
                          j*square_size:(j+1)*square_size] = 1.0

    elif pattern == 'gradient':
        # Horizontal gradient
        for i in range(size):
            image[:, i] = i / (size - 1)

    elif pattern == 'circle':
        # Circle in center
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] = 1.0

    elif pattern == 'bars':
        # Vertical bars
        bar_width = size // 16
        for i in range(0, size, 2 * bar_width):
            image[:, i:i+bar_width] = 1.0

    return image


# ============================================================================
# Color Space Conversions
# ============================================================================

def rgb_to_grayscale(rgb_image):
    """
    Convert RGB image to grayscale using standard weights

    I = 0.299*R + 0.587*G + 0.114*B

    Parameters:
    -----------
    rgb_image : ndarray
        RGB image of shape (H, W, 3)

    Returns:
    --------
    gray : ndarray
        Grayscale image of shape (H, W)
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be RGB image of shape (H, W, 3)")

    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(rgb_image, weights)

    return gray


def rgb_to_hsv(rgb_image):
    """
    Convert RGB to HSV color space

    H: Hue [0, 360]
    S: Saturation [0, 1]
    V: Value [0, 1]

    Parameters:
    -----------
    rgb_image : ndarray
        RGB image of shape (H, W, 3), values in [0, 1]

    Returns:
    --------
    hsv : ndarray
        HSV image of shape (H, W, 3)
    """
    rgb = rgb_image.copy()
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_val = np.max(rgb, axis=2)
    min_val = np.min(rgb, axis=2)
    delta = max_val - min_val

    # Value
    v = max_val

    # Saturation
    s = np.zeros_like(v)
    mask = max_val != 0
    s[mask] = delta[mask] / max_val[mask]

    # Hue
    h = np.zeros_like(v)

    # Red is max
    mask = (max_val == r) & (delta != 0)
    h[mask] = 60 * (((g[mask] - b[mask]) / delta[mask]) % 6)

    # Green is max
    mask = (max_val == g) & (delta != 0)
    h[mask] = 60 * (((b[mask] - r[mask]) / delta[mask]) + 2)

    # Blue is max
    mask = (max_val == b) & (delta != 0)
    h[mask] = 60 * (((r[mask] - g[mask]) / delta[mask]) + 4)

    hsv = np.stack([h, s, v], axis=2)
    return hsv


def hsv_to_rgb(hsv_image):
    """
    Convert HSV to RGB color space

    Parameters:
    -----------
    hsv_image : ndarray
        HSV image of shape (H, W, 3)

    Returns:
    --------
    rgb : ndarray
        RGB image of shape (H, W, 3), values in [0, 1]
    """
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    rgb_prime = np.zeros_like(hsv_image)

    # Different cases based on hue
    mask = (0 <= h) & (h < 60)
    rgb_prime[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=1)

    mask = (60 <= h) & (h < 120)
    rgb_prime[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=1)

    mask = (120 <= h) & (h < 180)
    rgb_prime[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=1)

    mask = (180 <= h) & (h < 240)
    rgb_prime[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=1)

    mask = (240 <= h) & (h < 300)
    rgb_prime[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=1)

    mask = (300 <= h) & (h <= 360)
    rgb_prime[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=1)

    rgb = rgb_prime + m[:, :, np.newaxis]
    return np.clip(rgb, 0, 1)


# ============================================================================
# 2D Convolution
# ============================================================================

def convolve2d_direct(image, kernel, boundary='zero'):
    """
    Direct 2D convolution implementation

    Parameters:
    -----------
    image : ndarray
        Input image of shape (M1, M2)
    kernel : ndarray
        Convolution kernel of shape (N1, N2)
    boundary : str
        'zero', 'replicate', 'reflect', or 'wrap'

    Returns:
    --------
    output : ndarray
        Convolved image
    """
    M1, M2 = image.shape
    N1, N2 = kernel.shape

    # Flip kernel for convolution
    kernel_flipped = np.flip(np.flip(kernel, 0), 1)

    # Padding for boundary handling
    pad1 = N1 // 2
    pad2 = N2 // 2

    # Pad image based on boundary mode
    if boundary == 'zero':
        padded = np.pad(image, ((pad1, pad1), (pad2, pad2)), mode='constant')
    elif boundary == 'replicate':
        padded = np.pad(image, ((pad1, pad1), (pad2, pad2)), mode='edge')
    elif boundary == 'reflect':
        padded = np.pad(image, ((pad1, pad1), (pad2, pad2)), mode='reflect')
    elif boundary == 'wrap':
        padded = np.pad(image, ((pad1, pad1), (pad2, pad2)), mode='wrap')
    else:
        raise ValueError(f"Unknown boundary mode: {boundary}")

    # Output image (same size as input)
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(M1):
        for j in range(M2):
            # Extract neighborhood
            neighborhood = padded[i:i+N1, j:j+N2]
            # Compute weighted sum
            output[i, j] = np.sum(neighborhood * kernel_flipped)

    return output


def convolve2d_fft(image, kernel):
    """
    FFT-based 2D convolution

    Much faster for large kernels

    Parameters:
    -----------
    image : ndarray
        Input image of shape (M1, M2)
    kernel : ndarray
        Convolution kernel of shape (N1, N2)

    Returns:
    --------
    output : ndarray
        Convolved image (same size as input)
    """
    M1, M2 = image.shape
    N1, N2 = kernel.shape

    # Output size for linear convolution
    output_shape = (M1 + N1 - 1, M2 + N2 - 1)

    # Zero-pad to power of 2 for efficiency
    fft_shape = [2**int(np.ceil(np.log2(s))) for s in output_shape]

    # Compute FFTs
    Image_fft = np.fft.fft2(image, s=fft_shape)
    Kernel_fft = np.fft.fft2(kernel, s=fft_shape)

    # Multiply in frequency domain
    Output_fft = Image_fft * Kernel_fft

    # Inverse FFT
    output_full = np.real(np.fft.ifft2(Output_fft))

    # Extract valid region (same size as input)
    start1 = N1 // 2
    start2 = N2 // 2
    output = output_full[start1:start1+M1, start2:start2+M2]

    return output


# ============================================================================
# 2D DFT (Separable Implementation)
# ============================================================================

def dft2d_separable(image):
    """
    Compute 2D DFT using separable property

    Apply 1D DFT to columns, then to rows

    Parameters:
    -----------
    image : ndarray
        Input image of shape (M1, M2)

    Returns:
    --------
    X : ndarray
        2D DFT of image (complex-valued)
    """
    # Apply 1D FFT to each column
    X_cols = np.fft.fft(image, axis=0)

    # Apply 1D FFT to each row of result
    X = np.fft.fft(X_cols, axis=1)

    return X


def idft2d_separable(X):
    """
    Compute inverse 2D DFT using separable property

    Parameters:
    -----------
    X : ndarray
        2D DFT (complex-valued)

    Returns:
    --------
    image : ndarray
        Reconstructed image
    """
    # Apply 1D IFFT to each row
    x_rows = np.fft.ifft(X, axis=1)

    # Apply 1D IFFT to each column
    image = np.fft.ifft(x_rows, axis=0)

    return np.real(image)


# ============================================================================
# Common Kernels
# ============================================================================

def box_kernel(size):
    """
    Box (averaging) filter kernel

    Parameters:
    -----------
    size : int
        Kernel size (size x size)

    Returns:
    --------
    kernel : ndarray
        Normalized box kernel
    """
    kernel = np.ones((size, size)) / (size * size)
    return kernel


def gaussian_kernel(size, sigma):
    """
    Gaussian filter kernel

    h[n1, n2] = (1/(2π σ²)) * exp(-(n1² + n2²)/(2σ²))

    Parameters:
    -----------
    size : int
        Kernel size (size x size), should be odd
    sigma : float
        Standard deviation

    Returns:
    --------
    kernel : ndarray
        Normalized Gaussian kernel
    """
    center = size // 2
    n1, n2 = np.mgrid[-center:center+1, -center:center+1]

    # Gaussian function
    kernel = np.exp(-(n1**2 + n2**2) / (2 * sigma**2))

    # Normalize
    kernel = kernel / np.sum(kernel)

    return kernel


def sharpening_kernel():
    """
    Basic sharpening kernel (Laplacian-based)

    Returns:
    --------
    kernel : ndarray
        3x3 sharpening kernel
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    return kernel


def sobel_kernels():
    """
    Sobel edge detection kernels

    Returns:
    --------
    sobel_x : ndarray
        Horizontal edge detection kernel
    sobel_y : ndarray
        Vertical edge detection kernel
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


def edge_magnitude(image):
    """
    Compute edge magnitude using Sobel operator

    Parameters:
    -----------
    image : ndarray
        Input image

    Returns:
    --------
    magnitude : ndarray
        Edge magnitude
    angle : ndarray
        Edge direction in radians
    """
    sobel_x, sobel_y = sobel_kernels()

    # Compute gradients
    Gx = convolve2d_fft(image, sobel_x)
    Gy = convolve2d_fft(image, sobel_y)

    # Magnitude and direction
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx)

    return magnitude, angle


# ============================================================================
# Visualization
# ============================================================================

def display_image(image, title='Image', cmap='gray'):
    """
    Display single image

    Parameters:
    -----------
    image : ndarray
        Image to display
    title : str
        Plot title
    cmap : str
        Colormap
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar(shrink=0.8, aspect=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_spectrum(image, title='Magnitude Spectrum'):
    """
    Display magnitude spectrum of image

    Parameters:
    -----------
    image : ndarray
        Input image
    title : str
        Plot title
    """
    # Compute 2D DFT
    X = np.fft.fft2(image)

    # Shift zero frequency to center
    X_centered = np.fft.fftshift(X)

    # Compute magnitude with log scale
    magnitude = np.log(1 + np.abs(X_centered))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_3d_surface(image, title='3D Surface', subsample=4):
    """
    Plot image as 3D surface

    Parameters:
    -----------
    image : ndarray
        Image to plot
    title : str
        Plot title
    subsample : int
        Subsampling factor for clarity
    """
    # Subsample for better visualization
    image_sub = image[::subsample, ::subsample]

    M1, M2 = image_sub.shape
    n1, n2 = np.meshgrid(np.arange(M2), np.arange(M1))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(n2, n1, image_sub, cmap=cm.viridis,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Column (n₂)')
    ax.set_ylabel('Row (n₁)')
    ax.set_zlabel('Intensity')
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def compare_images(images, titles, cmap='gray'):
    """
    Display multiple images side by side

    Parameters:
    -----------
    images : list of ndarray
        Images to display
    titles : list of str
        Titles for each image
    cmap : str
        Colormap
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    if n == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Examples
# ============================================================================

def example1_test_images():
    """
    Example 1: Create and display test images
    """
    print("Example 1: Test Image Generation")
    print("=" * 50)

    patterns = ['checkerboard', 'gradient', 'circle', 'bars']
    images = [create_test_image(256, pattern) for pattern in patterns]

    compare_images(images, [p.capitalize() for p in patterns])


def example2_color_spaces():
    """
    Example 2: Color space conversions
    """
    print("\nExample 2: Color Space Conversions")
    print("=" * 50)

    # Create simple RGB image
    size = 256
    rgb = np.zeros((size, size, 3))

    # Red gradient
    for i in range(size):
        rgb[:, i, 0] = i / (size - 1)

    # Green gradient
    for i in range(size):
        rgb[i, :, 1] = i / (size - 1)

    # Convert to grayscale
    gray = rgb_to_grayscale(rgb)

    # Convert to HSV
    hsv = rgb_to_hsv(rgb)

    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hsv[:, :, 0], cmap='hsv')
    axes[0, 2].set_title('Hue Channel')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(hsv[:, :, 1], cmap='gray')
    axes[1, 0].set_title('Saturation Channel')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hsv[:, :, 2], cmap='gray')
    axes[1, 1].set_title('Value Channel')
    axes[1, 1].axis('off')

    # Convert back to RGB
    rgb_reconstructed = hsv_to_rgb(hsv)
    axes[1, 2].imshow(rgb_reconstructed)
    axes[1, 2].set_title('HSV → RGB')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"RGB to grayscale conversion completed")
    print(f"RGB to HSV and back: max error = {np.max(np.abs(rgb - rgb_reconstructed)):.6f}")


def example3_convolution_comparison():
    """
    Example 3: Compare direct vs FFT-based convolution
    """
    print("\nExample 3: Convolution Methods Comparison")
    print("=" * 50)

    # Create test image
    image = create_test_image(256, 'circle')

    # Test with different kernel sizes
    kernel_sizes = [3, 5, 11, 21]

    print(f"\nImage size: {image.shape}")
    print(f"\nTiming comparison:")
    print(f"{'Kernel Size':<15} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for size in kernel_sizes:
        kernel = box_kernel(size)

        # Time direct method
        start = time.time()
        result_direct = convolve2d_direct(image, kernel, boundary='replicate')
        time_direct = (time.time() - start) * 1000

        # Time FFT method
        start = time.time()
        result_fft = convolve2d_fft(image, kernel)
        time_fft = (time.time() - start) * 1000

        speedup = time_direct / time_fft

        print(f"{size:>2d} × {size:<2d}        {time_direct:>10.2f}      {time_fft:>10.2f}      {speedup:>6.2f}x")

        # Verify results are similar
        error = np.max(np.abs(result_direct - result_fft))
        if error > 1e-3:
            print(f"  Warning: Results differ by {error:.6f}")

    # Display example with 11x11 kernel
    kernel = box_kernel(11)
    result = convolve2d_fft(image, kernel)

    compare_images([image, result], ['Original', '11×11 Box Filter'])


def example4_common_kernels():
    """
    Example 4: Apply common kernels
    """
    print("\nExample 4: Common Image Kernels")
    print("=" * 50)

    # Create test image
    image = create_test_image(256, 'checkerboard')

    # Apply different kernels
    box = convolve2d_fft(image, box_kernel(5))
    gaussian = convolve2d_fft(image, gaussian_kernel(11, 2.0))
    sharpened = convolve2d_fft(image, sharpening_kernel())

    # Edge detection
    edges, _ = edge_magnitude(image)

    images = [image, box, gaussian, sharpened, edges]
    titles = ['Original', 'Box 5×5', 'Gaussian σ=2', 'Sharpened', 'Edges']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


def example5_separable_dft():
    """
    Example 5: Demonstrate separable 2D DFT
    """
    print("\nExample 5: Separable 2D DFT")
    print("=" * 50)

    # Create test image
    image = create_test_image(128, 'bars')

    # Compute DFT using separable method
    X_sep = dft2d_separable(image)

    # Compute DFT using numpy (for comparison)
    X_np = np.fft.fft2(image)

    # Verify they're identical
    error = np.max(np.abs(X_sep - X_np))
    print(f"Max difference between separable and numpy DFT: {error:.10f}")

    # Reconstruct image
    image_reconstructed = idft2d_separable(X_sep)
    reconstruction_error = np.max(np.abs(image - image_reconstructed))
    print(f"Reconstruction error: {reconstruction_error:.10f}")

    # Display magnitude spectrum
    X_centered = np.fft.fftshift(X_sep)
    magnitude = np.log(1 + np.abs(X_centered))

    compare_images([image, magnitude], ['Original', 'Magnitude Spectrum (log)'])


def example6_boundary_modes():
    """
    Example 6: Compare boundary handling modes
    """
    print("\nExample 6: Boundary Handling Modes")
    print("=" * 50)

    # Create image with strong edge
    image = create_test_image(128, 'circle')

    # Apply large blur kernel with different boundary modes
    kernel = gaussian_kernel(31, 5.0)

    modes = ['zero', 'replicate', 'reflect', 'wrap']
    results = []

    for mode in modes:
        result = convolve2d_direct(image, kernel, boundary=mode)
        results.append(result)

    compare_images(results, [f'Boundary: {m}' for m in modes])


def example7_frequency_analysis():
    """
    Example 7: Frequency domain analysis of images
    """
    print("\nExample 7: Frequency Domain Analysis")
    print("=" * 50)

    # Create images with different frequency content
    low_freq = create_test_image(256, 'circle')  # Smooth
    high_freq = create_test_image(256, 'checkerboard')  # High detail

    # Compute spectra
    X_low = np.fft.fftshift(np.fft.fft2(low_freq))
    X_high = np.fft.fftshift(np.fft.fft2(high_freq))

    mag_low = np.log(1 + np.abs(X_low))
    mag_high = np.log(1 + np.abs(X_high))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(low_freq, cmap='gray')
    axes[0, 0].set_title('Low Frequency Content (Smooth)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mag_low, cmap='gray')
    axes[0, 1].set_title('Spectrum (Concentrated at Center)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(high_freq, cmap='gray')
    axes[1, 0].set_title('High Frequency Content (Detailed)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mag_high, cmap='gray')
    axes[1, 1].set_title('Spectrum (Spread Out)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("Low frequency image: Energy concentrated near DC component")
    print("High frequency image: Energy spread across frequency domain")


def run_all_examples():
    """
    Run all examples
    """
    print("Module 8: 2D Signals and Images")
    print("=" * 70)

    example1_test_images()
    example2_color_spaces()
    example3_convolution_comparison()
    example4_common_kernels()
    example5_separable_dft()
    example6_boundary_modes()
    example7_frequency_analysis()

    print("\n" + "=" * 70)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
