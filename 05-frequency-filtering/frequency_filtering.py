import numpy as np
import matplotlib.pyplot as plt


def frequency_domain_filter(x, H):
    """
    Apply filter in frequency domain.

    Args:
        x: Input signal
        H: Frequency response (DFT of impulse response)

    Returns:
        y: Filtered signal

    Note: Uses FFT for efficiency
    """
    # FFT of input
    X = np.fft.fft(x, n=len(H))

    # Multiply in frequency domain
    Y = X * H

    # IFFT to get output
    y = np.fft.ifft(Y).real

    return y


def design_lowpass_filter(N, cutoff_freq, filter_type='ideal'):
    """
    Design a lowpass filter.

    Args:
        N: Filter length (number of frequency bins)
        cutoff_freq: Normalized cutoff frequency (0 to 0.5)
        filter_type: 'ideal' or 'smooth'

    Returns:
        H: Frequency response (length N)
    """
    freqs = np.fft.fftfreq(N)

    if filter_type == 'ideal':
        # Ideal brick-wall filter
        H = np.abs(freqs) <= cutoff_freq
    elif filter_type == 'smooth':
        # Smooth transition using raised cosine
        transition_width = 0.05
        H = np.zeros(N)
        for i, f in enumerate(freqs):
            f_abs = abs(f)
            if f_abs <= cutoff_freq - transition_width:
                H[i] = 1.0
            elif f_abs >= cutoff_freq + transition_width:
                H[i] = 0.0
            else:
                # Cosine transition
                ratio = (f_abs - (cutoff_freq - transition_width)) / (2 * transition_width)
                H[i] = 0.5 * (1 + np.cos(np.pi * ratio))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return H


def design_highpass_filter(N, cutoff_freq, filter_type='ideal'):
    """
    Design a highpass filter.

    Args:
        N: Filter length
        cutoff_freq: Normalized cutoff frequency (0 to 0.5)
        filter_type: 'ideal' or 'smooth'

    Returns:
        H: Frequency response
    """
    # Highpass = 1 - Lowpass
    H_lp = design_lowpass_filter(N, cutoff_freq, filter_type)
    return 1.0 - H_lp


def design_bandpass_filter(N, f_low, f_high, filter_type='ideal'):
    """
    Design a bandpass filter.

    Args:
        N: Filter length
        f_low: Lower cutoff frequency (normalized)
        f_high: Upper cutoff frequency (normalized)
        filter_type: 'ideal' or 'smooth'

    Returns:
        H: Frequency response
    """
    freqs = np.fft.fftfreq(N)

    if filter_type == 'ideal':
        H = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    elif filter_type == 'smooth':
        # Product of two smooth transitions
        transition_width = 0.02
        H = np.zeros(N)

        for i, f in enumerate(freqs):
            f_abs = abs(f)

            # Lower edge
            if f_abs < f_low - transition_width:
                h_low = 0.0
            elif f_abs > f_low + transition_width:
                h_low = 1.0
            else:
                ratio = (f_abs - (f_low - transition_width)) / (2 * transition_width)
                h_low = 0.5 * (1 - np.cos(np.pi * ratio))

            # Upper edge
            if f_abs < f_high - transition_width:
                h_high = 1.0
            elif f_abs > f_high + transition_width:
                h_high = 0.0
            else:
                ratio = (f_abs - (f_high - transition_width)) / (2 * transition_width)
                h_high = 0.5 * (1 + np.cos(np.pi * ratio))

            H[i] = h_low * h_high
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return H.astype(float)


def design_bandstop_filter(N, f_low, f_high, filter_type='ideal'):
    """
    Design a bandstop (notch) filter.

    Args:
        N: Filter length
        f_low: Lower cutoff frequency
        f_high: Upper cutoff frequency
        filter_type: 'ideal' or 'smooth'

    Returns:
        H: Frequency response
    """
    # Bandstop = 1 - Bandpass
    H_bp = design_bandpass_filter(N, f_low, f_high, filter_type)
    return 1.0 - H_bp


def power_spectral_density(x):
    """
    Compute power spectral density (periodogram).

    Args:
        x: Input signal

    Returns:
        freqs: Frequency array
        psd: Power spectral density
    """
    N = len(x)
    X = np.fft.fft(x)

    # Periodogram: PSD = (1/N) * |X|^2
    psd = (1.0 / N) * np.abs(X) ** 2

    freqs = np.fft.fftfreq(N)

    return freqs, psd


def remove_frequency_component(x, freq_to_remove, bandwidth=0.01):
    """
    Remove a specific frequency component from signal.

    Args:
        x: Input signal
        freq_to_remove: Frequency to remove (normalized, 0 to 0.5)
        bandwidth: Width of notch filter

    Returns:
        y: Filtered signal
    """
    N = len(x)

    # Design notch filter
    H = design_bandstop_filter(N, freq_to_remove - bandwidth / 2,
                                freq_to_remove + bandwidth / 2,
                                filter_type='smooth')

    # Apply filter
    y = frequency_domain_filter(x, H)

    return y


def overlap_add_filter(x, h, block_size=None):
    """
    Filter long signal using overlap-add method.

    Args:
        x: Input signal (can be very long)
        h: Impulse response
        block_size: Size of blocks to process (default: 4 * len(h))

    Returns:
        y: Filtered signal
    """
    M = len(h)
    N_x = len(x)

    if block_size is None:
        block_size = 4 * M

    # FFT size: must be >= block_size + M - 1
    fft_size = 2 ** int(np.ceil(np.log2(block_size + M - 1)))

    # Precompute FFT of impulse response
    H = np.fft.fft(h, fft_size)

    # Output buffer
    y = np.zeros(N_x + M - 1)

    # Process blocks
    num_blocks = int(np.ceil(N_x / block_size))

    for i in range(num_blocks):
        # Extract block
        start = i * block_size
        end = min(start + block_size, N_x)
        x_block = x[start:end]

        # Zero-pad block
        x_padded = np.pad(x_block, (0, fft_size - len(x_block)))

        # FFT, multiply, IFFT
        X = np.fft.fft(x_padded)
        Y = X * H
        y_block = np.fft.ifft(Y).real

        # Add to output (overlap-add)
        y[start:start + len(y_block)] += y_block

    # Trim to correct length
    return y[:N_x + M - 1]


def compare_time_vs_freq_filtering(x, h):
    """
    Compare time-domain and frequency-domain filtering.

    Args:
        x: Input signal
        h: Impulse response

    Returns:
        results: Dictionary with timing and output comparison
    """
    import time

    # Time domain
    start = time.time()
    y_time = np.convolve(x, h, mode='full')
    time_domain = time.time() - start

    # Frequency domain
    start = time.time()
    N = len(x) + len(h) - 1
    fft_size = 2 ** int(np.ceil(np.log2(N)))
    X = np.fft.fft(x, fft_size)
    H_freq = np.fft.fft(h, fft_size)
    Y = X * H_freq
    y_freq = np.fft.ifft(Y).real[:N]
    freq_domain = time.time() - start

    # Verify they match
    match = np.allclose(y_time, y_freq)

    return {
        'time_domain_time': time_domain,
        'freq_domain_time': freq_domain,
        'speedup': time_domain / freq_domain,
        'outputs_match': match,
        'y_time': y_time,
        'y_freq': y_freq
    }


def plot_filter_response(H, Fs=1.0, title="Filter Response"):
    """
    Plot magnitude and phase response of a filter.

    Args:
        H: Frequency response (DFT)
        Fs: Sampling frequency
        title: Plot title
    """
    N = len(H)
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Take only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    H_pos = H[pos_mask]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Magnitude response (dB)
    mag_db = 20 * np.log10(np.abs(H_pos) + 1e-10)
    axes[0].plot(freqs_pos, mag_db, 'b', linewidth=2)
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'{title} - Magnitude Response')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3dB cutoff')
    axes[0].legend()

    # Phase response
    axes[1].plot(freqs_pos, np.angle(H_pos), 'r', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title(f'{title} - Phase Response')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()


if __name__ == "__main__":
    print("Module 5: Frequency Domain Filtering Examples\n")

    # Example 1: Filter design and visualization
    print("Example 1: Designing Filters\n")

    N = 256
    Fs = 1000  # 1 kHz sampling rate

    # Design different filter types
    H_lp = design_lowpass_filter(N, cutoff_freq=0.2, filter_type='smooth')
    H_hp = design_highpass_filter(N, cutoff_freq=0.2, filter_type='smooth')
    H_bp = design_bandpass_filter(N, f_low=0.1, f_high=0.3, filter_type='smooth')
    H_bs = design_bandstop_filter(N, f_low=0.2, f_high=0.25, filter_type='smooth')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    filters = [
        (H_lp, 'Lowpass (fc=0.2)', 0),
        (H_hp, 'Highpass (fc=0.2)', 1),
        (H_bp, 'Bandpass (0.1-0.3)', 2),
        (H_bs, 'Bandstop (0.2-0.25)', 3)
    ]

    freqs = np.fft.fftfreq(N, d=1/Fs)
    pos_mask = freqs >= 0

    for H, title, idx in filters:
        ax = axes.flat[idx]
        ax.plot(freqs[pos_mask], H[pos_mask], 'b', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|H(f)|')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.show()

    # Example 2: Noise removal
    print("\nExample 2: Removing High-Frequency Noise\n")

    # Create signal: 50 Hz sine + high-frequency noise
    Fs = 1000
    t = np.arange(0, 1, 1/Fs)
    signal_clean = np.sin(2 * np.pi * 50 * t)
    noise = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.random.randn(len(t))
    signal_noisy = signal_clean + noise

    # Design lowpass filter (cutoff at 100 Hz)
    N = len(signal_noisy)
    f_cutoff = 100 / Fs  # Normalized frequency
    H = design_lowpass_filter(N, f_cutoff, filter_type='smooth')

    # Filter signal
    signal_filtered = frequency_domain_filter(signal_noisy, H)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t[:200], signal_clean[:200], 'g', label='Clean signal (50 Hz)', linewidth=2)
    plt.ylabel('Amplitude')
    plt.title('Original Clean Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(t[:200], signal_noisy[:200], 'gray', label='Noisy signal', alpha=0.7)
    plt.ylabel('Amplitude')
    plt.title('Noisy Signal (50 Hz + 300 Hz + random noise)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(t[:200], signal_filtered[:200], 'b', label='Filtered signal', linewidth=2)
    plt.plot(t[:200], signal_clean[:200], 'g--', label='Original clean', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('After Lowpass Filtering (cutoff = 100 Hz)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 3: 60 Hz notch filter
    print("\nExample 3: Removing 60 Hz Power Line Interference\n")

    # Create signal with 60 Hz interference
    t = np.arange(0, 2, 1/Fs)
    signal = np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 120 * t)
    interference = 0.8 * np.sin(2 * np.pi * 60 * t)
    signal_corrupted = signal + interference

    # Remove 60 Hz component
    f_remove = 60 / Fs
    signal_clean = remove_frequency_component(signal_corrupted, f_remove, bandwidth=0.02)

    # Compute spectra
    freqs_orig, psd_orig = power_spectral_density(signal_corrupted)
    freqs_clean, psd_clean = power_spectral_density(signal_clean)

    # Plot spectra
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    pos_mask = (freqs_orig >= 0) & (freqs_orig <= 0.5)
    plt.semilogy(freqs_orig[pos_mask] * Fs, psd_orig[pos_mask], 'b', linewidth=2)
    plt.axvline(x=60, color='r', linestyle='--', label='60 Hz interference')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Original Signal Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(freqs_clean[pos_mask] * Fs, psd_clean[pos_mask], 'g', linewidth=2)
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.3, label='60 Hz removed')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('After Notch Filtering')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 4: Time vs Frequency domain performance
    print("\nExample 4: Performance Comparison\n")

    filter_lengths = [10, 50, 100, 200, 500]
    signal_length = 10000

    print(f"{'Filter Length':>15} | {'Time Domain':>15} | {'Freq Domain':>15} | {'Speedup':>10}")
    print("-" * 70)

    for M in filter_lengths:
        x = np.random.randn(signal_length)
        h = np.random.randn(M)

        results = compare_time_vs_freq_filtering(x, h)

        print(f"{M:15d} | {results['time_domain_time']:13.6f}s | "
              f"{results['freq_domain_time']:13.6f}s | "
              f"{results['speedup']:9.2f}x")

    # Example 5: Overlap-add for long signals
    print("\nExample 5: Overlap-Add Filtering\n")

    # Very long signal
    x_long = np.random.randn(100000)
    h = np.random.randn(200)

    # Compare direct vs overlap-add
    import time

    start = time.time()
    y_direct = np.convolve(x_long, h, mode='full')
    time_direct = time.time() - start

    start = time.time()
    y_overlap = overlap_add_filter(x_long, h, block_size=4096)
    time_overlap = time.time() - start

    print(f"Signal length: {len(x_long)}")
    print(f"Filter length: {len(h)}")
    print(f"\nDirect convolution:   {time_direct:.4f}s")
    print(f"Overlap-add:          {time_overlap:.4f}s")
    print(f"Speedup:              {time_direct / time_overlap:.2f}x")
    print(f"Outputs match:        {np.allclose(y_direct, y_overlap)}")

    print("\nModule 5 examples completed!")
