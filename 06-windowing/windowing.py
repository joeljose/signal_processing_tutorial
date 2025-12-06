import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def rectangular_window(N):
    """
    Rectangular (no) window.

    Args:
        N: Window length

    Returns:
        w: Window coefficients
    """
    return np.ones(N)


def hamming_window(N):
    """
    Hamming window.

    Args:
        N: Window length

    Returns:
        w: Window coefficients

    Formula: w[n] = 0.54 - 0.46*cos(2πn/(N-1))
    """
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def hann_window(N):
    """
    Hann (Hanning) window.

    Args:
        N: Window length

    Returns:
        w: Window coefficients

    Formula: w[n] = 0.5*(1 - cos(2πn/(N-1)))
    """
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))


def blackman_window(N):
    """
    Blackman window.

    Args:
        N: Window length

    Returns:
        w: Window coefficients

    Formula: w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
    """
    n = np.arange(N)
    return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) +
            0.08 * np.cos(4 * np.pi * n / (N - 1)))


def kaiser_window(N, beta):
    """
    Kaiser window.

    Args:
        N: Window length
        beta: Shape parameter (0 = rectangular, 5 ≈ Hamming, 8.6 ≈ Blackman)

    Returns:
        w: Window coefficients
    """
    return np.kaiser(N, beta)


def compare_windows(N=51):
    """
    Compare different window functions.

    Args:
        N: Window length

    Returns:
        Dictionary with window functions
    """
    windows = {
        'Rectangular': rectangular_window(N),
        'Hamming': hamming_window(N),
        'Hann': hann_window(N),
        'Blackman': blackman_window(N),
        'Kaiser (β=5)': kaiser_window(N, beta=5),
        'Kaiser (β=8.6)': kaiser_window(N, beta=8.6)
    }

    return windows


def window_frequency_response(w, nfft=2048):
    """
    Compute frequency response of a window function.

    Args:
        w: Window coefficients
        nfft: FFT size

    Returns:
        freqs: Normalized frequency array
        H: Frequency response (magnitude in dB)
    """
    # Compute FFT
    W = np.fft.fft(w, nfft)
    W = np.fft.fftshift(W)

    # Magnitude in dB
    H_db = 20 * np.log10(np.abs(W) / np.max(np.abs(W)) + 1e-10)

    # Frequency array
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft))

    return freqs, H_db


def demonstrate_spectral_leakage(f0=0.1, N=64, window_type='rectangular'):
    """
    Demonstrate spectral leakage with different windows.

    Args:
        f0: Signal frequency (normalized, 0 to 0.5)
        N: Signal length
        window_type: Type of window to apply

    Returns:
        freqs: Frequency array
        spectrum: Magnitude spectrum
    """
    # Generate sinusoid
    n = np.arange(N)
    x = np.cos(2 * np.pi * f0 * n)

    # Apply window
    if window_type == 'rectangular':
        w = rectangular_window(N)
    elif window_type == 'hamming':
        w = hamming_window(N)
    elif window_type == 'hann':
        w = hann_window(N)
    elif window_type == 'blackman':
        w = blackman_window(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    x_windowed = x * w

    # Compute spectrum
    X = np.fft.fft(x_windowed, n=512)
    freqs = np.fft.fftfreq(512)

    # Magnitude in dB
    spectrum = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-10)

    return freqs, spectrum


def demonstrate_zero_padding(x, padding_factors=[1, 2, 4, 8]):
    """
    Demonstrate effect of zero-padding.

    Args:
        x: Input signal
        padding_factors: List of padding factors

    Returns:
        Dictionary with results for each padding factor
    """
    N = len(x)
    results = {}

    for factor in padding_factors:
        N_fft = N * factor

        # Compute FFT with zero-padding
        X = np.fft.fft(x, n=N_fft)
        freqs = np.fft.fftfreq(N_fft)

        results[factor] = {
            'freqs': freqs,
            'magnitude': np.abs(X),
            'N_fft': N_fft,
            'num_bins': N_fft,
            'bin_spacing': 1.0 / N_fft  # Normalized frequency
        }

    return results


def fir_lowpass_window_method(cutoff_freq, N, window_type='hamming'):
    """
    Design FIR lowpass filter using window method.

    Args:
        cutoff_freq: Normalized cutoff frequency (0 to 0.5)
        N: Filter length (odd number recommended)
        window_type: Type of window

    Returns:
        h: FIR filter coefficients
    """
    # Ensure N is odd for symmetric Type I filter
    if N % 2 == 0:
        N += 1

    # Ideal impulse response (sinc function)
    n = np.arange(N)
    n_center = (N - 1) / 2

    # Avoid division by zero
    h_ideal = np.zeros(N)
    for i in range(N):
        if i == n_center:
            h_ideal[i] = 2 * cutoff_freq
        else:
            h_ideal[i] = np.sin(2 * np.pi * cutoff_freq * (i - n_center)) / (np.pi * (i - n_center))

    # Apply window
    if window_type == 'rectangular':
        w = rectangular_window(N)
    elif window_type == 'hamming':
        w = hamming_window(N)
    elif window_type == 'hann':
        w = hann_window(N)
    elif window_type == 'blackman':
        w = blackman_window(N)
    elif window_type.startswith('kaiser'):
        beta = float(window_type.split('_')[1]) if '_' in window_type else 5
        w = kaiser_window(N, beta)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Windowed filter
    h = h_ideal * w

    return h


def plot_window_comparison(windows_dict, N):
    """
    Plot comparison of window functions.

    Args:
        windows_dict: Dictionary of window functions
        N: Window length
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    n = np.arange(N)

    # Time domain
    for name, w in windows_dict.items():
        axes[0].plot(n, w, label=name, linewidth=2)

    axes[0].set_xlabel('Sample n')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Window Functions - Time Domain')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Frequency domain
    for name, w in windows_dict.items():
        freqs, H_db = window_frequency_response(w)
        axes[1].plot(freqs, H_db, label=name, linewidth=2)

    axes[1].set_xlabel('Normalized Frequency')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Window Functions - Frequency Response')
    axes[1].set_ylim([-120, 5])
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=-3, color='r', linestyle='--', alpha=0.3, label='-3dB')

    plt.tight_layout()


def window_normalization_factor(w):
    """
    Compute normalization factors for a window.

    Args:
        w: Window coefficients

    Returns:
        Dictionary with normalization factors
    """
    # Amplitude correction (for coherent signals)
    S1 = np.sum(w)
    amp_correction = len(w) / S1 if S1 != 0 else 1.0

    # Power correction (for incoherent signals)
    S2 = np.sum(w ** 2)
    power_correction = len(w) / S2 if S2 != 0 else 1.0

    return {
        'amplitude_correction': amp_correction,
        'power_correction': power_correction,
        'S1': S1,
        'S2': S2
    }


if __name__ == "__main__":
    print("Module 6: Windowing and Spectral Leakage Examples\n")

    # Example 1: Window function comparison
    print("Example 1: Comparing Window Functions\n")

    N = 51
    windows = compare_windows(N)

    plot_window_comparison(windows, N)
    plt.show()

    # Print window characteristics
    print(f"{'Window':20} | {'Peak Side Lobe':>15} | {'Main Lobe Width':>16}")
    print("-" * 60)

    characteristics = {
        'Rectangular': (-13, '4π/N'),
        'Hann': (-32, '8π/N'),
        'Hamming': (-43, '8π/N'),
        'Blackman': (-58, '12π/N')
    }

    for name, (peak_sl, mlw) in characteristics.items():
        print(f"{name:20} | {peak_sl:13} dB | {mlw:>16}")

    # Example 2: Spectral leakage demonstration
    print("\n" + "="*60)
    print("Example 2: Spectral Leakage with Different Windows\n")

    f0 = 0.123  # Not exactly on a DFT bin
    N_signal = 64

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window_types = ['rectangular', 'hamming', 'hann', 'blackman']

    for idx, wtype in enumerate(window_types):
        freqs, spectrum = demonstrate_spectral_leakage(f0, N_signal, wtype)

        ax = axes.flat[idx]
        pos_mask = (freqs >= 0) & (freqs <= 0.5)
        ax.plot(freqs[pos_mask], spectrum[pos_mask], linewidth=2)
        ax.axvline(x=f0, color='r', linestyle='--', label=f'True freq = {f0}')
        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'{wtype.capitalize()} Window')
        ax.set_ylim([-80, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    print("Notice how side lobes decrease with better windows!")
    print("Trade-off: Main lobe gets wider.")

    # Example 3: Zero-padding demonstration
    print("\n" + "="*60)
    print("Example 3: Zero-Padding Effects\n")

    # Short signal
    N = 32
    n = np.arange(N)
    f_signal = 0.15
    x = np.cos(2 * np.pi * f_signal * n)

    # Apply Hamming window
    w = hamming_window(N)
    x_windowed = x * w

    # Zero-padding
    results = demonstrate_zero_padding(x_windowed, [1, 2, 4, 8])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (factor, data) in enumerate(results.items()):
        ax = axes.flat[idx]
        freqs = data['freqs']
        mag = data['magnitude']

        pos_mask = (freqs >= 0) & (freqs <= 0.5)
        ax.stem(freqs[pos_mask], mag[pos_mask], basefmt=' ')
        ax.axvline(x=f_signal, color='r', linestyle='--',
                   label=f'True freq = {f_signal}')
        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Zero-padding ×{factor} (N_FFT = {data["N_fft"]})')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"Original signal length: {N}")
    print(f"Frequency resolution: {1.0/N:.4f} (independent of zero-padding!)\n")

    for factor, data in results.items():
        print(f"Padding ×{factor}: {data['num_bins']} bins, "
              f"bin spacing = {data['bin_spacing']:.6f}")

    print("\nZero-padding gives more bins but same fundamental resolution!")

    # Example 4: FIR filter design using windows
    print("\n" + "="*60)
    print("Example 4: FIR Lowpass Filter Design\n")

    cutoff = 0.2  # Normalized cutoff frequency
    filter_length = 51

    window_types_fir = ['rectangular', 'hamming', 'hann', 'blackman']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, wtype in enumerate(window_types_fir):
        # Design filter
        h = fir_lowpass_window_method(cutoff, filter_length, wtype)

        # Compute frequency response
        w_freq, H = signal.freqz(h, worN=2048)
        H_db = 20 * np.log10(np.abs(H) + 1e-10)

        ax = axes.flat[idx]
        ax.plot(w_freq / np.pi, H_db, linewidth=2)
        ax.axvline(x=cutoff * 2, color='r', linestyle='--',
                   label=f'Cutoff = {cutoff}')
        ax.axhline(y=-3, color='g', linestyle='--', alpha=0.5,
                   label='-3dB')
        ax.set_xlabel('Normalized Frequency (×π rad/sample)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'FIR Lowpass - {wtype.capitalize()} Window')
        ax.set_ylim([-100, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"Filter length: {filter_length}")
    print(f"Cutoff frequency: {cutoff}\n")

    print("Observations:")
    print("- Rectangular: Sharpest transition, high ripple in stopband")
    print("- Hamming: Good stopband attenuation (~-53 dB)")
    print("- Blackman: Excellent stopband (~-74 dB), wider transition")

    # Example 5: Window normalization
    print("\n" + "="*60)
    print("Example 5: Window Normalization Factors\n")

    N = 100
    print(f"{'Window':15} | {'S1 (Sum)':>12} | {'S2 (Energy)':>14} | "
          f"{'Amp Corr':>10} | {'Power Corr':>11}")
    print("-" * 75)

    for name, w in compare_windows(N).items():
        norm = window_normalization_factor(w)
        print(f"{name:15} | {norm['S1']:12.2f} | {norm['S2']:14.2f} | "
              f"{norm['amplitude_correction']:10.4f} | "
              f"{norm['power_correction']:11.4f}")

    print("\nUse amplitude correction for coherent signals (known frequency)")
    print("Use power correction for incoherent signals (noise, random)")

    # Example 6: Kaiser window parameter sweep
    print("\n" + "="*60)
    print("Example 6: Kaiser Window - Adjustable Trade-off\n")

    N = 51
    betas = [0, 3, 5, 8.6, 12]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for beta in betas:
        w = kaiser_window(N, beta)
        plt.plot(np.arange(N), w, label=f'β = {beta}', linewidth=2)
    plt.xlabel('Sample n')
    plt.ylabel('Amplitude')
    plt.title('Kaiser Window - Time Domain')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for beta in betas:
        w = kaiser_window(N, beta)
        freqs, H_db = window_frequency_response(w)
        plt.plot(freqs, H_db, label=f'β = {beta}', linewidth=2)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.title('Kaiser Window - Frequency Response')
    plt.xlim([-0.3, 0.3])
    plt.ylim([-120, 5])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Kaiser window allows continuous trade-off:")
    print("- β = 0: Rectangular (narrow main lobe, high side lobes)")
    print("- β = 5: Similar to Hamming")
    print("- β = 8.6: Similar to Blackman")
    print("- β > 10: Even lower side lobes, wider main lobe")

    print("\nModule 6 examples completed!")
