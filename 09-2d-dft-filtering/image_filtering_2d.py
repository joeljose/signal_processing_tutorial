"""
Module 9: 2D DFT and Image Filtering
Comprehensive implementation of frequency domain image filtering techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

# ============================================================================
# Frequency Domain Filter Design
# ============================================================================

def create_distance_matrix(M, N, centered=True):
    """
    Create distance matrix from center

    Parameters:
    -----------
    M, N : int
        Image dimensions
    centered : bool
        If True, distances from center; if False, from (0,0)

    Returns:
    --------
    D : ndarray
        Distance matrix of shape (M, N)
    """
    if centered:
        k = np.arange(M) - M // 2
        l = np.arange(N) - N // 2
    else:
        k = np.arange(M)
        l = np.arange(N)

    kk, ll = np.meshgrid(l, k)  # Note: meshgrid swaps dimensions
    D = np.sqrt(kk**2 + ll**2)

    return D


# ============================================================================
# Ideal Filters
# ============================================================================

def ideal_lowpass_filter(M, N, D0):
    """
    Ideal lowpass filter

    H[k,l] = 1 if D(k,l) <= D0, else 0

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D0 : float
        Cutoff frequency

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    H = np.zeros((M, N))
    H[D <= D0] = 1.0
    return np.fft.ifftshift(H)  # Convert to standard FFT layout


def ideal_highpass_filter(M, N, D0):
    """
    Ideal highpass filter

    H[k,l] = 0 if D(k,l) <= D0, else 1

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D0 : float
        Cutoff frequency

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    return 1.0 - ideal_lowpass_filter(M, N, D0)


def ideal_bandpass_filter(M, N, D1, D2):
    """
    Ideal bandpass filter

    Passes frequencies in range [D1, D2]

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D1, D2 : float
        Inner and outer cutoff frequencies

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    H = np.zeros((M, N))
    mask = (D >= D1) & (D <= D2)
    H[mask] = 1.0
    return np.fft.ifftshift(H)


def ideal_bandstop_filter(M, N, D1, D2):
    """
    Ideal bandstop (notch) filter

    Blocks frequencies in range [D1, D2]

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D1, D2 : float
        Inner and outer cutoff frequencies

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    return 1.0 - ideal_bandpass_filter(M, N, D1, D2)


# ============================================================================
# Butterworth Filters
# ============================================================================

def butterworth_lowpass_filter(M, N, D0, n=2):
    """
    Butterworth lowpass filter

    H[k,l] = 1 / (1 + (D/D0)^(2n))

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D0 : float
        Cutoff frequency (at H = 0.5)
    n : int
        Filter order (higher = sharper transition)

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    H = 1.0 / (1.0 + (D / (D0 + 1e-10))**(2 * n))
    return np.fft.ifftshift(H)


def butterworth_highpass_filter(M, N, D0, n=2):
    """
    Butterworth highpass filter

    H[k,l] = 1 / (1 + (D0/D)^(2n))

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D0 : float
        Cutoff frequency
    n : int
        Filter order

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    # Avoid division by zero at DC
    D[M//2, N//2] = 1e-10
    H = 1.0 / (1.0 + (D0 / D)**(2 * n))
    return np.fft.ifftshift(H)


def butterworth_bandpass_filter(M, N, D1, D2, n=2):
    """
    Butterworth bandpass filter

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    D1, D2 : float
        Inner and outer cutoff frequencies
    n : int
        Filter order

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D_center = (D1 + D2) / 2
    bandwidth = D2 - D1

    D = create_distance_matrix(M, N, centered=True)
    H = 1.0 / (1.0 + ((D**2 - D_center**2) / (D * bandwidth + 1e-10))**(2 * n))

    return np.fft.ifftshift(H)


# ============================================================================
# Gaussian Filters
# ============================================================================

def gaussian_lowpass_filter(M, N, sigma):
    """
    Gaussian lowpass filter

    H[k,l] = exp(-D²/(2σ²))

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    sigma : float
        Standard deviation (controls width)

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    H = np.exp(-D**2 / (2 * sigma**2))
    return np.fft.ifftshift(H)


def gaussian_highpass_filter(M, N, sigma):
    """
    Gaussian highpass filter

    H[k,l] = 1 - exp(-D²/(2σ²))

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    sigma : float
        Standard deviation

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    return 1.0 - gaussian_lowpass_filter(M, N, sigma)


def gaussian_bandpass_filter(M, N, sigma1, sigma2):
    """
    Gaussian bandpass filter (difference of Gaussians)

    H = exp(-D²/(2σ₂²)) - exp(-D²/(2σ₁²))

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    sigma1, sigma2 : float
        Inner and outer standard deviations (sigma1 < sigma2)

    Returns:
    --------
    H : ndarray
        Filter of shape (M, N)
    """
    D = create_distance_matrix(M, N, centered=True)
    H = np.exp(-D**2 / (2 * sigma2**2)) - np.exp(-D**2 / (2 * sigma1**2))
    H = np.maximum(H, 0)  # Ensure non-negative
    return np.fft.ifftshift(H)


# ============================================================================
# Frequency Domain Filtering
# ============================================================================

def frequency_filter(image, filter_func, *args, **kwargs):
    """
    Apply frequency domain filter to image

    Parameters:
    -----------
    image : ndarray
        Input image
    filter_func : function
        Filter function that returns H[k,l]
    *args, **kwargs : additional arguments for filter_func

    Returns:
    --------
    filtered : ndarray
        Filtered image
    H : ndarray
        Filter used
    """
    M, N = image.shape

    # Compute FFT
    F = np.fft.fft2(image)

    # Create filter
    H = filter_func(M, N, *args, **kwargs)

    # Apply filter
    G = F * H

    # Inverse FFT
    filtered = np.real(np.fft.ifft2(G))

    return filtered, H


# ============================================================================
# Homomorphic Filtering
# ============================================================================

def homomorphic_filter(M, N, gamma_low=0.5, gamma_high=2.0, c=1.0, D0=30):
    """
    Homomorphic filter for illumination normalization

    H[k,l] = (γ_H - γ_L) * (1 - exp(-c·D²/D0²)) + γ_L

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    gamma_low : float
        Attenuation for low frequencies (< 1)
    gamma_high : float
        Amplification for high frequencies (> 1)
    c : float
        Controls sharpness of transition
    D0 : float
        Cutoff frequency

    Returns:
    --------
    H : ndarray
        Homomorphic filter
    """
    D = create_distance_matrix(M, N, centered=True)
    H = (gamma_high - gamma_low) * (1 - np.exp(-c * (D**2) / (D0**2))) + gamma_low
    return np.fft.ifftshift(H)


def apply_homomorphic_filter(image, gamma_low=0.5, gamma_high=2.0, c=1.0, D0=30):
    """
    Apply homomorphic filtering to enhance image

    Steps:
    1. log(image)
    2. FFT
    3. Apply homomorphic filter
    4. IFFT
    5. exp(result)

    Parameters:
    -----------
    image : ndarray
        Input image (values > 0)
    gamma_low, gamma_high, c, D0 : homomorphic filter parameters

    Returns:
    --------
    enhanced : ndarray
        Enhanced image
    """
    # Ensure positive values
    image = np.maximum(image, 1e-10)

    # Log transform
    log_image = np.log(image)

    # FFT
    F = np.fft.fft2(log_image)

    # Create and apply filter
    M, N = image.shape
    H = homomorphic_filter(M, N, gamma_low, gamma_high, c, D0)
    G = F * H

    # IFFT
    log_filtered = np.real(np.fft.ifft2(G))

    # Exponential
    enhanced = np.exp(log_filtered)

    return enhanced


# ============================================================================
# Notch Filters
# ============================================================================

def create_notch_filter(M, N, notch_positions, radius=5):
    """
    Create notch filter to remove specific frequencies

    Parameters:
    -----------
    M, N : int
        Filter dimensions
    notch_positions : list of tuples
        List of (k, l) positions to notch out
    radius : float
        Radius of notch

    Returns:
    --------
    H : ndarray
        Notch filter
    """
    H = np.ones((M, N))

    k_grid = np.arange(M) - M // 2
    l_grid = np.arange(N) - N // 2
    kk, ll = np.meshgrid(l_grid, k_grid)

    for (k0, l0) in notch_positions:
        # Distance from notch center
        D1 = np.sqrt((kk - k0)**2 + (ll - l0)**2)
        D2 = np.sqrt((kk + k0)**2 + (ll + l0)**2)  # Symmetric position

        # Zero out notch
        H[D1 <= radius] = 0
        H[D2 <= radius] = 0

    return np.fft.ifftshift(H)


# ============================================================================
# Separable Filtering
# ============================================================================

def separable_gaussian_filter(image, sigma):
    """
    Apply Gaussian filter using separable implementation

    Faster than full 2D convolution!

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
    # Create 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= np.sum(kernel_1d)

    # Apply to rows
    filtered = ndimage.convolve1d(image, kernel_1d, axis=1, mode='reflect')

    # Apply to columns
    filtered = ndimage.convolve1d(filtered, kernel_1d, axis=0, mode='reflect')

    return filtered


# ============================================================================
# Analysis Tools
# ============================================================================

def analyze_frequency_content(image):
    """
    Analyze frequency content of image

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    M, N = image.shape

    # Compute FFT
    F = np.fft.fft2(image)
    F_centered = np.fft.fftshift(F)

    # Magnitude and phase
    magnitude = np.abs(F_centered)
    phase = np.angle(F_centered)

    # Total energy
    energy_total = np.sum(magnitude**2)

    # Compute energy in different frequency bands
    D = create_distance_matrix(M, N, centered=True)

    results = {
        'dc_component': F[0, 0],
        'energy_total': energy_total,
        'magnitude': magnitude,
        'phase': phase
    }

    # Energy in concentric bands
    for cutoff in [10, 30, 50, 100]:
        energy_low = np.sum(magnitude[D <= cutoff]**2)
        ratio = energy_low / energy_total
        results[f'energy_ratio_D{cutoff}'] = ratio

    return results


def swap_magnitude_phase(image1, image2):
    """
    Demonstrate importance of phase

    Create images by swapping magnitude and phase

    Parameters:
    -----------
    image1, image2 : ndarray
        Two images

    Returns:
    --------
    mag1_phase2 : ndarray
        Magnitude of image1, phase of image2
    mag2_phase1 : ndarray
        Magnitude of image2, phase of image1
    """
    # FFTs
    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)

    # Extract magnitude and phase
    mag1 = np.abs(F1)
    mag2 = np.abs(F2)
    phase1 = np.angle(F1)
    phase2 = np.angle(F2)

    # Reconstruct with swapped components
    F_mag1_phase2 = mag1 * np.exp(1j * phase2)
    F_mag2_phase1 = mag2 * np.exp(1j * phase1)

    # Inverse FFT
    mag1_phase2 = np.real(np.fft.ifft2(F_mag1_phase2))
    mag2_phase1 = np.real(np.fft.ifft2(F_mag2_phase1))

    return mag1_phase2, mag2_phase1


# ============================================================================
# Visualization
# ============================================================================

def display_filter_response(H, title='Filter Frequency Response'):
    """
    Display filter frequency response

    Parameters:
    -----------
    H : ndarray
        Filter in standard FFT layout
    title : str
        Plot title
    """
    # Shift to centered layout for display
    H_centered = np.fft.fftshift(H)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 3D view
    M, N = H.shape
    k = np.arange(M) - M // 2
    l = np.arange(N) - N // 2
    kk, ll = np.meshgrid(l, k)

    # Subsample for visualization
    step = max(1, M // 50)
    kk_sub = kk[::step, ::step]
    ll_sub = ll[::step, ::step]
    H_sub = H_centered[::step, ::step]

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(ll_sub, kk_sub, H_sub, cmap='viridis')
    ax1.set_xlabel('Frequency l')
    ax1.set_ylabel('Frequency k')
    ax1.set_zlabel('H[k,l]')
    ax1.set_title(f'{title} (3D)')

    # 2D view
    axes[1].imshow(H_centered, cmap='gray', extent=[-N//2, N//2, -M//2, M//2])
    axes[1].set_xlabel('Frequency l')
    axes[1].set_ylabel('Frequency k')
    axes[1].set_title(f'{title} (2D)')
    axes[1].colorbar()

    plt.tight_layout()
    plt.show()


def compare_filters(image, filters, titles):
    """
    Compare multiple filtering results

    Parameters:
    -----------
    image : ndarray
        Original image
    filters : list of ndarray
        List of filters
    titles : list of str
        List of titles
    """
    n = len(filters) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Filtered images
    for i, (H, title) in enumerate(zip(filters, titles)):
        filtered, _ = frequency_filter(image, lambda M, N: H)

        axes[i+1].imshow(filtered, cmap='gray')
        axes[i+1].set_title(title)
        axes[i+1].axis('off')

    # Hide extra subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Examples
# ============================================================================

def create_test_image(size=256, pattern='checkerboard'):
    """Create test images (imported from Module 8 for convenience)"""
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

    elif pattern == 'bars':
        bar_width = size // 16
        for i in range(0, size, 2 * bar_width):
            image[:, i:i+bar_width] = 1.0

    return image


def example1_filter_comparison():
    """
    Example 1: Compare ideal, Butterworth, and Gaussian filters
    """
    print("Example 1: Filter Type Comparison")
    print("=" * 50)

    size = 256
    D0 = 30

    # Create filters
    H_ideal = ideal_lowpass_filter(size, size, D0)
    H_butterworth = butterworth_lowpass_filter(size, size, D0, n=2)
    H_gaussian = gaussian_lowpass_filter(size, size, sigma=D0/2)

    print(f"Filter size: {size}×{size}")
    print(f"Cutoff frequency: {D0}")

    # Display filter responses
    for H, name in [(H_ideal, 'Ideal'),
                     (H_butterworth, 'Butterworth (n=2)'),
                     (H_gaussian, 'Gaussian (σ=15)')]:
        display_filter_response(H, f'{name} Lowpass Filter')

    # Apply to test image
    image = create_test_image(size, 'checkerboard')

    filters = [H_ideal, H_butterworth, H_gaussian]
    titles = ['Ideal LP', 'Butterworth LP', 'Gaussian LP']

    compare_filters(image, filters, titles)

    print("Note: Ideal filter shows ringing artifacts (Gibbs phenomenon)")
    print("Butterworth and Gaussian filters have smoother transitions")


def example2_lowpass_highpass():
    """
    Example 2: Lowpass vs Highpass filtering
    """
    print("\nExample 2: Lowpass vs Highpass Filtering")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'circle')

    # Add some noise
    image_noisy = image + 0.1 * np.random.randn(size, size)
    image_noisy = np.clip(image_noisy, 0, 1)

    # Lowpass filter (smoothing, noise removal)
    lowpass, H_lp = frequency_filter(image_noisy, gaussian_lowpass_filter, sigma=20)

    # Highpass filter (edge enhancement)
    highpass, H_hp = frequency_filter(image, gaussian_highpass_filter, sigma=20)

    # Rescale highpass for display
    highpass_display = (highpass - highpass.min()) / (highpass.max() - highpass.min())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image_noisy, cmap='gray')
    axes[0, 0].set_title('Original (Noisy)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.fft.fftshift(H_lp), cmap='gray')
    axes[0, 1].set_title('Lowpass Filter')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(lowpass, cmap='gray')
    axes[0, 2].set_title('Smoothed Result')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].set_title('Original (Clean)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.fft.fftshift(H_hp), cmap='gray')
    axes[1, 1].set_title('Highpass Filter')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(highpass_display, cmap='gray')
    axes[1, 2].set_title('Edges Enhanced')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Lowpass: Removes noise, smooths image")
    print("Highpass: Enhances edges, removes low-frequency content")


def example3_butterworth_order():
    """
    Example 3: Effect of Butterworth filter order
    """
    print("\nExample 3: Butterworth Filter Order")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'checkerboard')
    D0 = 30

    orders = [1, 2, 4, 8]

    print(f"Cutoff frequency D0 = {D0}")
    print(f"Testing orders: {orders}")

    filters = []
    titles = []

    for n in orders:
        H = butterworth_lowpass_filter(size, size, D0, n=n)
        filters.append(H)
        titles.append(f'Order n={n}')

    compare_filters(image, filters, titles)

    print("\nAs order increases:")
    print("  - Transition becomes sharper")
    print("  - Approaches ideal filter")
    print("  - More ringing artifacts appear")


def example4_homomorphic_filtering():
    """
    Example 4: Homomorphic filtering for illumination correction
    """
    print("\nExample 4: Homomorphic Filtering")
    print("=" * 50)

    size = 256

    # Create image with uneven illumination
    image = create_test_image(size, 'checkerboard')

    # Add illumination gradient
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    illumination = 0.3 + 0.7 * (xx + yy) / 2

    image_uneven = image * illumination
    image_uneven = np.clip(image_uneven, 0.01, 1.0)  # Ensure positive

    # Apply homomorphic filter
    enhanced = apply_homomorphic_filter(image_uneven,
                                        gamma_low=0.5,
                                        gamma_high=2.0,
                                        c=1.0,
                                        D0=30)

    # Normalize for display
    enhanced_display = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_uneven, cmap='gray')
    axes[0].set_title('Uneven Illumination')
    axes[0].axis('off')

    axes[1].imshow(illumination, cmap='gray')
    axes[1].set_title('Illumination Component')
    axes[1].axis('off')

    axes[2].imshow(enhanced_display, cmap='gray')
    axes[2].set_title('After Homomorphic Filtering')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Homomorphic filtering:")
    print("  - Reduces illumination variation")
    print("  - Enhances details")
    print("  - Normalizes brightness across image")


def example5_notch_filter():
    """
    Example 5: Notch filter for periodic noise removal
    """
    print("\nExample 5: Notch Filtering")
    print("=" * 50)

    size = 256
    image = create_test_image(size, 'circle')

    # Add periodic noise (simulating interference)
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    # Two periodic components
    noise = 0.3 * np.sin(2 * np.pi * 40 * xx / size) + \
            0.3 * np.sin(2 * np.pi * 40 * yy / size)

    image_noisy = image + noise
    image_noisy = np.clip(image_noisy, 0, 1)

    # Analyze spectrum to find noise peaks
    F = np.fft.fftshift(np.fft.fft2(image_noisy))
    spectrum = np.log(1 + np.abs(F))

    # Create notch filter (manually specify notch positions)
    # For this synthetic example, we know the frequencies
    notch_positions = [(40, 0), (0, 40)]  # Adjusted for centering
    H = create_notch_filter(size, size, notch_positions, radius=5)

    # Apply filter
    filtered, _ = frequency_filter(image_noisy, lambda M, N: H)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image_noisy, cmap='gray')
    axes[0, 0].set_title('Noisy Image (Periodic Interference)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(spectrum, cmap='gray')
    axes[0, 1].set_title('Spectrum (Shows Noise Peaks)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(np.fft.fftshift(H), cmap='gray')
    axes[1, 0].set_title('Notch Filter')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(filtered, cmap='gray')
    axes[1, 1].set_title('Filtered Result')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("Notch filter removes specific frequency components")
    print("Useful for eliminating periodic interference")


def example6_phase_importance():
    """
    Example 6: Demonstrate importance of phase
    """
    print("\nExample 6: Phase vs Magnitude Importance")
    print("=" * 50)

    size = 256

    # Create two distinct images
    image1 = create_test_image(size, 'circle')
    image2 = create_test_image(size, 'checkerboard')

    # Swap magnitude and phase
    mag1_phase2, mag2_phase1 = swap_magnitude_phase(image1, image2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 0].set_title('Image 1 (Circle)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title('Image 2 (Checkerboard)')
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    axes[1, 0].imshow(mag1_phase2, cmap='gray')
    axes[1, 0].set_title('Mag(Image1) + Phase(Image2)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mag2_phase1, cmap='gray')
    axes[1, 1].set_title('Mag(Image2) + Phase(Image1)')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Key observation:")
    print("  - Magnitude + Phase of different images")
    print("  - Result looks more like the phase source!")
    print("  - Phase contains structural information")
    print("  - Magnitude contains energy distribution")


def example7_separable_filtering():
    """
    Example 7: Separable filter efficiency
    """
    print("\nExample 7: Separable Filtering Efficiency")
    print("=" * 50)

    sizes = [128, 256, 512]
    sigma = 5.0

    print(f"Gaussian blur with σ = {sigma}")
    print(f"\n{'Size':<10} {'Frequency (ms)':<20} {'Separable (ms)':<20} {'Speedup':<10}")
    print("-" * 70)

    for size in sizes:
        image = create_test_image(size, 'circle')

        # Frequency domain method
        start = time.time()
        filtered_freq, _ = frequency_filter(image, gaussian_lowpass_filter, sigma=sigma*2)
        time_freq = (time.time() - start) * 1000

        # Separable method
        start = time.time()
        filtered_sep = separable_gaussian_filter(image, sigma)
        time_sep = (time.time() - start) * 1000

        speedup = time_freq / time_sep

        print(f"{size}×{size:<4} {time_freq:>15.2f}     {time_sep:>15.2f}     {speedup:>6.2f}x")

        # Verify results are similar
        error = np.max(np.abs(filtered_freq - filtered_sep))
        if error > 0.01:
            print(f"  Warning: Methods differ by {error:.6f}")

    print("\nSeparable filtering is much faster for spatial domain operations!")


def run_all_examples():
    """
    Run all examples
    """
    print("Module 9: 2D DFT and Image Filtering")
    print("=" * 70)

    example1_filter_comparison()
    example2_lowpass_highpass()
    example3_butterworth_order()
    example4_homomorphic_filtering()
    example5_notch_filter()
    example6_phase_importance()
    example7_separable_filtering()

    print("\n" + "=" * 70)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
