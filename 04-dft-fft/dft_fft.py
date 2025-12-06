import numpy as np
import matplotlib.pyplot as plt
import time


def dft_direct(x):
    """
    Compute DFT using direct method (slow, O(N^2)).

    Args:
        x: Input signal (length N)

    Returns:
        X: DFT of x (length N)

    Formula:
        X[k] = Σ x[n] * e^(-j2πkn/N) for n=0 to N-1
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    # Twiddle factor matrix: W_N^(kn) = e^(-j2πkn/N)
    W = np.exp(-2j * np.pi * k * n / N)

    # Matrix multiplication
    X = np.dot(W, x)

    return X


def fft_recursive(x):
    """
    Compute FFT using recursive Cooley-Tukey algorithm.

    Args:
        x: Input signal (length N, must be power of 2)

    Returns:
        X: FFT of x

    Complexity: O(N log N)
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)

    # Base case: 1-point DFT
    if N == 1:
        return x

    # Check if N is power of 2
    if N % 2 != 0:
        raise ValueError("Length of x must be a power of 2")

    # Divide: split into even and odd indices
    x_even = fft_recursive(x[::2])
    x_odd = fft_recursive(x[1::2])

    # Twiddle factors
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)

    # Combine: butterfly operation
    X = np.concatenate([
        x_even + W * x_odd,
        x_even - W * x_odd
    ])

    return X


def fft_iterative(x):
    """
    Compute FFT using iterative Cooley-Tukey algorithm (more efficient).

    Args:
        x: Input signal (length N, must be power of 2)

    Returns:
        X: FFT of x
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)

    # Check if N is power of 2
    if N & (N - 1) != 0:
        raise ValueError("Length of x must be a power of 2")

    # Bit-reversal permutation
    X = bit_reverse_copy(x)

    # Compute FFT in-place
    num_stages = int(np.log2(N))

    for s in range(1, num_stages + 1):
        m = 2 ** s  # Size of DFT in this stage
        W_m = np.exp(-2j * np.pi / m)  # Twiddle factor for this stage

        for k in range(0, N, m):
            W = 1
            for j in range(m // 2):
                t = W * X[k + j + m // 2]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + m // 2] = u - t
                W = W * W_m

    return X


def bit_reverse_copy(x):
    """
    Bit-reverse permutation for iterative FFT.

    Args:
        x: Input array

    Returns:
        Bit-reversed copy of x
    """
    N = len(x)
    num_bits = int(np.log2(N))
    reversed_x = np.zeros_like(x)

    for i in range(N):
        # Reverse bits of i
        reversed_i = int(bin(i)[2:].zfill(num_bits)[::-1], 2)
        reversed_x[reversed_i] = x[i]

    return reversed_x


def idft(X):
    """
    Compute inverse DFT.

    Args:
        X: Frequency domain signal

    Returns:
        x: Time domain signal
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)

    # IDFT = conj(DFT(conj(X))) / N
    x = np.conj(dft_direct(np.conj(X))) / N

    return x


def ifft(X):
    """
    Compute inverse FFT.

    Args:
        X: Frequency domain signal (length must be power of 2)

    Returns:
        x: Time domain signal
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)

    # IFFT = conj(FFT(conj(X))) / N
    x = np.conj(fft_recursive(np.conj(X))) / N

    return x


def circular_convolution(x, h):
    """
    Compute circular convolution of two sequences.

    Args:
        x, h: Input sequences (same length N)

    Returns:
        y: Circular convolution x ⊛ h

    Formula:
        y[n] = Σ x[m] * h[(n-m) mod N] for m=0 to N-1
    """
    N = len(x)
    if len(h) != N:
        raise ValueError("Sequences must have the same length")

    y = np.zeros(N)
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]

    return y


def circular_convolution_fft(x, h):
    """
    Compute circular convolution using FFT (fast).

    Args:
        x, h: Input sequences

    Returns:
        y: Circular convolution
    """
    # Pad to same length
    N = max(len(x), len(h))
    x_padded = np.pad(x, (0, N - len(x)))
    h_padded = np.pad(h, (0, N - len(h)))

    # FFT, multiply, IFFT
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    Y = X * H
    y = np.fft.ifft(Y).real

    return y


def linear_convolution_fft(x, h):
    """
    Compute linear convolution using FFT with zero-padding.

    Args:
        x, h: Input sequences

    Returns:
        y: Linear convolution (same as np.convolve(x, h))
    """
    # Output length for linear convolution
    L = len(x) + len(h) - 1

    # Zero-pad to next power of 2 for efficiency
    N = 2 ** int(np.ceil(np.log2(L)))

    # Zero-pad both sequences
    x_padded = np.pad(x, (0, N - len(x)))
    h_padded = np.pad(h, (0, N - len(h)))

    # FFT, multiply, IFFT
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    Y = X * H
    y = np.fft.ifft(Y).real[:L]  # Take only first L points

    return y


def compare_dft_fft(N_values):
    """
    Compare computation time of DFT vs FFT.

    Args:
        N_values: List of signal lengths to test

    Returns:
        times_dft, times_fft: Arrays of computation times
    """
    times_dft = []
    times_fft = []

    for N in N_values:
        # Generate random signal
        x = np.random.randn(N)

        # Time DFT
        start = time.time()
        X_dft = dft_direct(x)
        time_dft = time.time() - start
        times_dft.append(time_dft)

        # Time FFT
        start = time.time()
        X_fft = np.fft.fft(x)
        time_fft = time.time() - start
        times_fft.append(time_fft)

        # Verify they're the same
        if not np.allclose(X_dft, X_fft):
            print(f"Warning: DFT and FFT don't match for N={N}")

    return np.array(times_dft), np.array(times_fft)


def plot_dft_spectrum(x, Fs=1.0, title="DFT Spectrum"):
    """
    Plot magnitude and phase spectrum of a signal.

    Args:
        x: Time domain signal
        Fs: Sampling frequency (Hz)
        title: Plot title
    """
    N = len(x)
    X = np.fft.fft(x)

    # Frequency bins
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Take only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    X_pos = X[pos_mask]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Magnitude spectrum
    axes[0].stem(freqs_pos, np.abs(X_pos), basefmt=' ')
    axes[0].set_ylabel('|X[k]|')
    axes[0].set_title(f'{title} - Magnitude Spectrum')
    axes[0].grid(True, alpha=0.3)

    # Phase spectrum
    axes[1].stem(freqs_pos, np.angle(X_pos), basefmt=' ')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('∠X[k] (radians)')
    axes[1].set_title(f'{title} - Phase Spectrum')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()


if __name__ == "__main__":
    print("Module 4: DFT and FFT Examples\n")

    # Example 1: Simple DFT computation
    print("Example 1: DFT of a Simple Signal\n")

    x = np.array([1, 2, 3, 4])
    print(f"Signal x = {x}")

    X_dft = dft_direct(x)
    X_fft = fft_recursive(x)
    X_numpy = np.fft.fft(x)

    print(f"\nDFT (direct):    {X_dft}")
    print(f"FFT (recursive): {X_fft}")
    print(f"NumPy FFT:       {X_numpy}")
    print(f"\nAll methods agree: {np.allclose(X_dft, X_fft) and np.allclose(X_dft, X_numpy)}")

    # Example 2: Performance comparison
    print("\n" + "="*60)
    print("Example 2: DFT vs FFT Performance Comparison\n")

    N_values = [2**i for i in range(4, 11)]  # 16, 32, ..., 1024
    times_dft, times_fft = compare_dft_fft(N_values)

    print(f"{'N':>6} | {'DFT Time':>12} | {'FFT Time':>12} | {'Speedup':>8}")
    print("-" * 55)
    for N, t_dft, t_fft in zip(N_values, times_dft, times_fft):
        speedup = t_dft / t_fft
        print(f"{N:6d} | {t_dft:10.6f}s | {t_fft:10.6f}s | {speedup:7.1f}x")

    # Plot performance
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.loglog(N_values, times_dft, 'bo-', label='DFT O(N²)', linewidth=2)
    plt.loglog(N_values, times_fft, 'rs-', label='FFT O(N log N)', linewidth=2)
    plt.xlabel('Signal Length N')
    plt.ylabel('Computation Time (seconds)')
    plt.title('DFT vs FFT Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogx(N_values, times_dft / times_fft, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Signal Length N')
    plt.ylabel('Speedup (DFT time / FFT time)')
    plt.title('FFT Speedup Factor')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 3: Circular vs Linear Convolution
    print("\n" + "="*60)
    print("Example 3: Circular vs Linear Convolution\n")

    x = np.array([1, 2, 3])
    h = np.array([0.5, 0.5])

    # Pad to same length for circular convolution
    N = len(x)
    h_padded = np.pad(h, (0, N - len(h)))

    y_circular = circular_convolution(x, h_padded)
    y_linear = np.convolve(x, h)

    print(f"x = {x}")
    print(f"h = {h}")
    print(f"\nCircular convolution: {y_circular}")
    print(f"Linear convolution:   {y_linear}")
    print("\nNote: Different results due to wrap-around in circular convolution!")

    # Example 4: Linear convolution via FFT
    print("\n" + "="*60)
    print("Example 4: Linear Convolution via FFT\n")

    y_fft = linear_convolution_fft(x, h)
    y_direct = np.convolve(x, h)

    print(f"Linear conv (FFT):    {y_fft}")
    print(f"Linear conv (direct): {y_direct}")
    print(f"Match: {np.allclose(y_fft, y_direct)}")

    # Example 5: Frequency spectrum of composite signal
    print("\n" + "="*60)
    print("Example 5: Frequency Spectrum Analysis\n")

    # Create signal with multiple frequency components
    Fs = 1000  # Sampling rate: 1000 Hz
    T = 1.0    # Duration: 1 second
    N = int(Fs * T)
    t = np.arange(N) / Fs

    # Signal: 50 Hz + 120 Hz + 200 Hz
    f1, f2, f3 = 50, 120, 200
    x = (np.sin(2*np.pi*f1*t) +
         0.5*np.sin(2*np.pi*f2*t) +
         0.3*np.sin(2*np.pi*f3*t))

    print(f"Signal: {f1} Hz + {f2} Hz + {f3} Hz")
    print(f"Sampling rate: {Fs} Hz")
    print(f"Number of samples: {N}")
    print(f"Frequency resolution: {Fs/N:.2f} Hz")

    plot_dft_spectrum(x, Fs, "Composite Signal")
    plt.show()

    # Example 6: Zero-padding effect
    print("\n" + "="*60)
    print("Example 6: Effect of Zero-Padding on Frequency Resolution\n")

    # Short signal
    Fs = 100
    t_short = np.arange(32) / Fs
    x_short = np.sin(2*np.pi*10*t_short)

    # Zero-padded versions
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, N_fft in enumerate([32, 128, 512]):
        X = np.fft.fft(x_short, N_fft)
        freqs = np.fft.fftfreq(N_fft, d=1/Fs)

        pos_mask = freqs >= 0
        axes[idx].stem(freqs[pos_mask], np.abs(X[pos_mask]), basefmt=' ')
        axes[idx].set_ylabel('|X[k]|')
        axes[idx].set_title(f'N_fft = {N_fft}, Δf = {Fs/N_fft:.2f} Hz')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 50])

    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

    print("\nModule 4 examples completed!")
