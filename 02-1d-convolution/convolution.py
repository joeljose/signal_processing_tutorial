import numpy as np
import matplotlib.pyplot as plt
import time


def convolve_direct(x, h):
    """
    Compute convolution using direct method (definition).

    Args:
        x: Input signal (length M)
        h: Impulse response (length N)

    Returns:
        y: Output signal (length M + N - 1)
    """
    M = len(x)
    N = len(h)
    y_length = M + N - 1

    # Initialize output
    y = np.zeros(y_length)

    # Direct convolution
    for n in range(y_length):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += h[k] * x[n - k]

    return y


def convolve_fft(x, h):
    """
    Compute convolution using FFT method.

    Args:
        x: Input signal
        h: Impulse response

    Returns:
        y: Output signal (same length as direct convolution)
    """
    # Determine output length
    output_length = len(x) + len(h) - 1

    # Zero-pad to next power of 2 for efficiency (optional)
    fft_length = 2 ** int(np.ceil(np.log2(output_length)))

    # Compute FFT of both signals
    X = np.fft.fft(x, fft_length)
    H = np.fft.fft(h, fft_length)

    # Multiply in frequency domain
    Y = X * H

    # Inverse FFT
    y = np.fft.ifft(Y)

    # Take real part and trim to correct length
    return np.real(y[:output_length])


def moving_average_filter(signal, window_size):
    """
    Apply moving average filter using convolution.

    Args:
        signal: Input signal
        window_size: Number of points to average

    Returns:
        Filtered signal
    """
    h = np.ones(window_size) / window_size
    return np.convolve(signal, h, mode='same')


def first_difference(signal):
    """
    Compute first difference (simple high-pass filter).

    Args:
        signal: Input signal

    Returns:
        First difference
    """
    h = np.array([1, -1])
    return np.convolve(signal, h, mode='same')


def compare_convolution_methods(x, h):
    """
    Compare direct and FFT convolution methods.

    Args:
        x: Input signal
        h: Impulse response

    Returns:
        Dictionary with results and timing info
    """
    # Direct method
    start = time.time()
    y_direct = convolve_direct(x, h)
    time_direct = time.time() - start

    # FFT method
    start = time.time()
    y_fft = convolve_fft(x, h)
    time_fft = time.time() - start

    # NumPy's convolve (for comparison)
    start = time.time()
    y_numpy = np.convolve(x, h)
    time_numpy = time.time() - start

    # Verify they're all equal
    are_equal = np.allclose(y_direct, y_fft) and np.allclose(y_direct, y_numpy)

    return {
        'direct': y_direct,
        'fft': y_fft,
        'numpy': y_numpy,
        'time_direct': time_direct,
        'time_fft': time_fft,
        'time_numpy': time_numpy,
        'are_equal': are_equal
    }


def verify_convolution_properties():
    """
    Verify mathematical properties of convolution.
    """
    # Generate test signals
    x = np.array([1, 2, 3, 4])
    h1 = np.array([0.5, 0.5])
    h2 = np.array([1, -1])

    print("Verifying Convolution Properties\n")
    print("=" * 50)

    # 1. Commutativity: x * h = h * x
    lhs = np.convolve(x, h1)
    rhs = np.convolve(h1, x)
    print("\n1. Commutativity: x * h = h * x")
    print(f"   x * h1 = {lhs}")
    print(f"   h1 * x = {rhs}")
    print(f"   Equal: {np.allclose(lhs, rhs)}")

    # 2. Identity: x * δ = x
    delta = np.array([1])  # Unit impulse
    result = np.convolve(x, delta, mode='same')
    print("\n2. Identity: x * δ = x")
    print(f"   x = {x}")
    print(f"   x * δ = {result}")
    print(f"   Equal: {np.allclose(x, result)}")

    # 3. Distributivity: x * (h1 + h2) = x * h1 + x * h2
    # Pad to same length for addition
    max_len = max(len(h1), len(h2))
    h1_padded = np.pad(h1, (0, max_len - len(h1)))
    h2_padded = np.pad(h2, (0, max_len - len(h2)))

    lhs = np.convolve(x, h1_padded + h2_padded)
    rhs = np.convolve(x, h1_padded) + np.convolve(x, h2_padded)
    print("\n3. Distributivity: x * (h1 + h2) = x * h1 + x * h2")
    print(f"   Equal: {np.allclose(lhs, rhs)}")

    print("\n" + "=" * 50)


def plot_convolution_example(x, h, title="Convolution Example"):
    """
    Visualize convolution operation.

    Args:
        x: Input signal
        h: Impulse response
        title: Plot title
    """
    y = np.convolve(x, h)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot input
    axes[0].stem(x, basefmt=" ")
    axes[0].set_ylabel('x[n]')
    axes[0].set_title('Input Signal')
    axes[0].grid(True, alpha=0.3)

    # Plot impulse response
    axes[1].stem(h, basefmt=" ")
    axes[1].set_ylabel('h[n]')
    axes[1].set_title('Impulse Response')
    axes[1].grid(True, alpha=0.3)

    # Plot output
    axes[2].stem(y, basefmt=" ")
    axes[2].set_ylabel('y[n]')
    axes[2].set_xlabel('n')
    axes[2].set_title(f'Output: y[n] = x[n] * h[n] (length = {len(y)})')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()


if __name__ == "__main__":
    print("Module 2: 1D Convolution Examples\n")

    # Example 1: Simple convolution
    print("Example 1: Simple Convolution")
    print("-" * 50)
    x = np.array([1, 2, 3])
    h = np.array([0.5, 0.5])

    y = np.convolve(x, h)
    print(f"x = {x}")
    print(f"h = {h}")
    print(f"y = x * h = {y}")
    print(f"Output length: {len(y)} (input: {len(x)}, filter: {len(h)})")

    plot_convolution_example(x, h, "Simple Convolution Example")
    plt.show()

    # Example 2: Verify properties
    verify_convolution_properties()

    # Example 3: Moving average (noise reduction)
    print("\nExample 3: Noise Reduction with Moving Average")
    print("-" * 50)

    # Create noisy signal
    n = np.arange(100)
    true_signal = np.sin(2 * np.pi * 0.05 * n)
    noise = 0.3 * np.random.randn(len(n))
    noisy_signal = true_signal + noise

    # Apply filters of different sizes
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(n, true_signal, 'g-', label='True Signal', linewidth=2)
    plt.plot(n, noisy_signal, 'gray', alpha=0.5, label='Noisy Signal')
    plt.ylabel('Amplitude')
    plt.title('Original Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(n, true_signal, 'g-', label='True Signal', linewidth=2, alpha=0.5)

    for window in [5, 10, 20]:
        filtered = moving_average_filter(noisy_signal, window)
        plt.plot(n, filtered, label=f'{window}-point MA', linewidth=1.5)

    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Filtered Signals (Moving Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 4: Performance comparison
    print("\nExample 4: Performance Comparison")
    print("-" * 50)

    for N in [100, 500, 1000]:
        x = np.random.randn(N)
        h = np.random.randn(50)

        results = compare_convolution_methods(x, h)

        print(f"\nSignal length: {N}, Filter length: 50")
        print(f"  Direct method:  {results['time_direct']*1000:.3f} ms")
        print(f"  FFT method:     {results['time_fft']*1000:.3f} ms")
        print(f"  NumPy convolve: {results['time_numpy']*1000:.3f} ms")
        print(f"  All equal: {results['are_equal']}")

    print("\nModule 2 examples completed!")
