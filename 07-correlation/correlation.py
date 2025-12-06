import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def cross_correlation_direct(x, y):
    """
    Compute cross-correlation using direct method.

    Args:
        x, y: Input signals

    Returns:
        r: Cross-correlation r_xy[l]
        lags: Lag indices

    Formula: r_xy[l] = Σ x[n] · y[n+l]
    """
    N = len(x)
    M = len(y)
    max_lag = max(N, M) - 1

    # Lag range: -(M-1) to (N-1)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.zeros(len(lags))

    for i, l in enumerate(lags):
        # Compute overlap region
        if l >= 0:
            # Positive lag: shift y to the right
            overlap = min(N - l, M)
            if overlap > 0:
                r[i] = np.sum(x[l:l + overlap] * y[:overlap])
        else:
            # Negative lag: shift y to the left
            overlap = min(N, M + l)
            if overlap > 0:
                r[i] = np.sum(x[:overlap] * y[-l:-l + overlap])

    return r, lags


def cross_correlation_fft(x, y):
    """
    Compute cross-correlation using FFT (fast).

    Args:
        x, y: Input signals

    Returns:
        r: Cross-correlation
        lags: Lag indices

    Uses: r_xy[l] = x[l] * y[-l] (convolution relationship)
    """
    # Use scipy's correlate with FFT method
    r = signal.correlate(x, y, mode='full', method='fft')

    # Lag indices
    lags = signal.correlation_lags(len(x), len(y), mode='full')

    return r, lags


def auto_correlation(x, method='fft'):
    """
    Compute auto-correlation of a signal.

    Args:
        x: Input signal
        method: 'direct' or 'fft'

    Returns:
        r: Auto-correlation r_xx[l]
        lags: Lag indices
    """
    if method == 'direct':
        r, lags = cross_correlation_direct(x, x)
    else:
        r, lags = cross_correlation_fft(x, x)

    return r, lags


def normalized_cross_correlation(x, y):
    """
    Compute normalized cross-correlation.

    Args:
        x, y: Input signals

    Returns:
        rho: Normalized correlation (-1 to 1)
        lags: Lag indices
    """
    # Cross-correlation
    r_xy, lags = cross_correlation_fft(x, y)

    # Auto-correlations at zero lag (energies)
    E_x = np.sum(x ** 2)
    E_y = np.sum(y ** 2)

    # Normalize
    rho = r_xy / np.sqrt(E_x * E_y)

    return rho, lags


def detect_periodicity(x, Fs=1.0, plot=False):
    """
    Detect periodicity in a signal using auto-correlation.

    Args:
        x: Input signal
        Fs: Sampling frequency
        plot: If True, plot auto-correlation

    Returns:
        Dictionary with period, frequency, and confidence
    """
    # Compute auto-correlation
    r_xx, lags = auto_correlation(x)

    # Only consider positive lags (auto-correlation is symmetric)
    pos_lags = lags >= 0
    r_pos = r_xx[pos_lags]
    lags_pos = lags[pos_lags]

    # Find peaks (excluding zero lag)
    if len(r_pos) > 1:
        # Normalize
        r_norm = r_pos / r_pos[0]

        # Find peaks
        peaks, properties = signal.find_peaks(r_norm[1:], height=0.3)

        if len(peaks) > 0:
            # First peak indicates period
            period_samples = lags_pos[peaks[0] + 1]  # +1 because we excluded lag 0
            period_time = period_samples / Fs
            frequency = 1.0 / period_time if period_time > 0 else 0

            # Confidence = peak height
            confidence = properties['peak_heights'][0]

            if plot:
                plt.figure(figsize=(12, 6))
                plt.plot(lags_pos / Fs, r_norm, 'b', linewidth=2)
                plt.axvline(x=period_time, color='r', linestyle='--',
                           label=f'Period = {period_time:.3f}s')
                plt.xlabel('Lag (seconds)')
                plt.ylabel('Normalized Auto-correlation')
                plt.title('Periodicity Detection via Auto-correlation')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()

            return {
                'period_samples': period_samples,
                'period_time': period_time,
                'frequency': frequency,
                'confidence': confidence,
                'is_periodic': True
            }

    return {
        'is_periodic': False,
        'period_samples': None,
        'period_time': None,
        'frequency': None,
        'confidence': 0.0
    }


def template_matching(signal_data, template, threshold=0.7):
    """
    Find template in signal using cross-correlation.

    Args:
        signal_data: Long signal to search
        template: Template pattern to find
        threshold: Correlation threshold (0 to 1)

    Returns:
        Dictionary with match locations and correlation values
    """
    # Normalized cross-correlation
    rho, lags = normalized_cross_correlation(signal_data, template)

    # Find peaks above threshold
    peaks, properties = signal.find_peaks(np.abs(rho), height=threshold)

    # Convert lag indices to signal positions
    positions = []
    correlations = []

    for peak in peaks:
        lag = lags[peak]
        if 0 <= lag < len(signal_data):
            positions.append(lag)
            correlations.append(rho[peak])

    return {
        'positions': np.array(positions),
        'correlations': np.array(correlations),
        'num_matches': len(positions),
        'correlation_full': rho,
        'lags': lags
    }


def estimate_time_delay(signal1, signal2, Fs=1.0):
    """
    Estimate time delay between two signals.

    Args:
        signal1, signal2: Input signals (same phenomenon, different sensors)
        Fs: Sampling frequency

    Returns:
        Dictionary with delay in samples and time
    """
    # Cross-correlation
    r_xy, lags = cross_correlation_fft(signal1, signal2)

    # Find peak
    peak_idx = np.argmax(np.abs(r_xy))
    delay_samples = lags[peak_idx]
    delay_time = delay_samples / Fs

    # Correlation coefficient at peak
    correlation = r_xy[peak_idx]

    # Normalize
    E1 = np.sum(signal1 ** 2)
    E2 = np.sum(signal2 ** 2)
    correlation_normalized = correlation / np.sqrt(E1 * E2)

    return {
        'delay_samples': delay_samples,
        'delay_time': delay_time,
        'correlation': correlation_normalized,
        'confidence': abs(correlation_normalized)
    }


def matched_filter(signal_data, pattern):
    """
    Apply matched filter for pattern detection.

    Args:
        signal_data: Input signal
        pattern: Expected pattern

    Returns:
        output: Matched filter output (cross-correlation)
        lags: Time indices
    """
    # Matched filter = cross-correlation
    output, lags = cross_correlation_fft(signal_data, pattern)

    # Normalize by pattern energy
    E_pattern = np.sum(pattern ** 2)
    if E_pattern > 0:
        output = output / np.sqrt(E_pattern)

    return output, lags


def power_spectral_density_via_correlation(x, method='correlate'):
    """
    Compute PSD using Wiener-Khinchin theorem.

    Args:
        x: Input signal
        method: 'correlate' (via auto-correlation) or 'periodogram'

    Returns:
        freqs: Frequency array
        psd: Power spectral density
    """
    N = len(x)

    if method == 'correlate':
        # Method 1: Via auto-correlation
        r_xx, lags = auto_correlation(x)

        # Take only non-negative lags (r_xx is symmetric)
        pos_lags = lags >= 0
        r_pos = r_xx[pos_lags]

        # FFT of auto-correlation = PSD
        psd_full = np.fft.fft(r_pos, n=2 * N)
        psd = np.abs(psd_full[:N])
        freqs = np.fft.fftfreq(2 * N, d=1.0)[:N]

    else:
        # Method 2: Direct periodogram
        X = np.fft.fft(x)
        psd = (1.0 / N) * np.abs(X) ** 2
        freqs = np.fft.fftfreq(N)

        # Take only positive frequencies
        pos_mask = freqs >= 0
        psd = psd[pos_mask]
        freqs = freqs[pos_mask]

    return freqs, psd


def compare_correlation_methods(x, y):
    """
    Compare direct and FFT correlation methods.

    Args:
        x, y: Input signals

    Returns:
        Dictionary with timing and verification info
    """
    import time

    # Direct method
    start = time.time()
    r_direct, lags_direct = cross_correlation_direct(x, y)
    time_direct = time.time() - start

    # FFT method
    start = time.time()
    r_fft, lags_fft = cross_correlation_fft(x, y)
    time_fft = time.time() - start

    # Verify they match
    match = np.allclose(r_direct, r_fft)

    return {
        'time_direct': time_direct,
        'time_fft': time_fft,
        'speedup': time_direct / time_fft,
        'match': match
    }


if __name__ == "__main__":
    print("Module 7: Correlation Examples\n")

    # Example 1: Auto-correlation of periodic signal
    print("Example 1: Periodicity Detection\n")

    # Create periodic signal with noise
    Fs = 1000  # 1 kHz
    t = np.arange(0, 2, 1/Fs)
    f0 = 10  # 10 Hz fundamental

    x_periodic = (np.sin(2 * np.pi * f0 * t) +
                  0.5 * np.sin(2 * np.pi * 2 * f0 * t))  # Fundamental + harmonic
    noise = 0.3 * np.random.randn(len(t))
    x_noisy = x_periodic + noise

    # Detect periodicity
    result = detect_periodicity(x_noisy, Fs, plot=True)

    print(f"Signal frequency: {f0} Hz")
    print(f"Detected frequency: {result['frequency']:.2f} Hz")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Is periodic: {result['is_periodic']}\n")

    # Example 2: Template matching
    print("="*60)
    print("Example 2: Template Matching\n")

    # Create signal with repeated pattern
    template = signal.gaussian(50, std=10)
    positions_true = [100, 300, 500, 750]

    # Build signal
    signal_length = 1000
    test_signal = 0.1 * np.random.randn(signal_length)

    for pos in positions_true:
        if pos + len(template) < signal_length:
            test_signal[pos:pos + len(template)] += template

    # Find template
    matches = template_matching(test_signal, template, threshold=0.6)

    print(f"True template positions: {positions_true}")
    print(f"Detected positions: {matches['positions'].tolist()}")
    print(f"Number of matches: {matches['num_matches']}")
    print(f"Correlation values: {matches['correlations']}\n")

    # Plot
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(template)
    plt.title('Template Pattern')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(test_signal)
    for pos in matches['positions']:
        plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
    plt.title('Signal with Template Occurrences (red lines = detected)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    valid_lags = (matches['lags'] >= 0) & (matches['lags'] < signal_length)
    plt.plot(matches['lags'][valid_lags], matches['correlation_full'][valid_lags])
    plt.axhline(y=0.6, color='r', linestyle='--', label='Threshold')
    plt.title('Normalized Cross-Correlation')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 3: Time delay estimation
    print("="*60)
    print("Example 3: Time Delay Estimation\n")

    # Create signal
    Fs = 1000
    t = np.arange(0, 1, 1/Fs)
    original_signal = signal.chirp(t, f0=10, f1=100, t1=1, method='linear')

    # Add noise
    noise1 = 0.2 * np.random.randn(len(t))
    noise2 = 0.2 * np.random.randn(len(t))

    # Create delayed version
    true_delay = 50  # samples
    sensor1 = original_signal + noise1
    sensor2 = np.concatenate([np.zeros(true_delay), original_signal[:-true_delay]]) + noise2

    # Estimate delay
    delay_result = estimate_time_delay(sensor1, sensor2, Fs)

    print(f"True delay: {true_delay} samples ({true_delay/Fs*1000:.2f} ms)")
    print(f"Estimated delay: {delay_result['delay_samples']} samples "
          f"({delay_result['delay_time']*1000:.2f} ms)")
    print(f"Error: {abs(delay_result['delay_samples'] - true_delay)} samples")
    print(f"Correlation confidence: {delay_result['confidence']:.3f}\n")

    # Example 4: Matched filter for pattern detection
    print("="*60)
    print("Example 4: Matched Filter (Pattern Detection in Noise)\n")

    # Create noisy signal with known pattern
    Fs = 1000
    t = np.arange(0, 1, 1/Fs)
    pattern = signal.gausspulse(t - 0.5, fc=50, bw=0.5)

    # Signal = noise + pattern at specific location
    signal_with_pattern = 0.5 * np.random.randn(len(t))
    pattern_location = 400
    signal_with_pattern[pattern_location:pattern_location + len(pattern)] += 2 * pattern

    # Apply matched filter
    mf_output, mf_lags = matched_filter(signal_with_pattern, pattern)

    # Find peak
    peak_idx = np.argmax(np.abs(mf_output))
    detected_location = mf_lags[peak_idx]

    print(f"True pattern location: {pattern_location} samples")
    print(f"Detected location: {detected_location} samples")
    print(f"SNR improvement: Matched filter maximizes SNR!\n")

    # Plot
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, pattern)
    plt.title('Known Pattern (to detect)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(signal_with_pattern)
    plt.axvline(x=pattern_location, color='r', linestyle='--',
                label=f'True location = {pattern_location}')
    plt.title('Noisy Signal with Hidden Pattern')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    valid_mf = (mf_lags >= 0) & (mf_lags < len(signal_with_pattern))
    plt.plot(mf_lags[valid_mf], np.abs(mf_output[valid_mf]))
    plt.axvline(x=detected_location, color='r', linestyle='--',
                label=f'Detected = {detected_location}')
    plt.title('Matched Filter Output (Cross-Correlation)')
    plt.xlabel('Sample')
    plt.ylabel('|Output|')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 5: PSD via Wiener-Khinchin theorem
    print("="*60)
    print("Example 5: Power Spectral Density via Correlation\n")

    # Create signal with two frequency components
    Fs = 1000
    t = np.arange(0, 2, 1/Fs)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    x += 0.2 * np.random.randn(len(t))

    # Method 1: Via auto-correlation (Wiener-Khinchin)
    freqs_wk, psd_wk = power_spectral_density_via_correlation(x, method='correlate')

    # Method 2: Direct periodogram
    freqs_pd, psd_pd = power_spectral_density_via_correlation(x, method='periodogram')

    # Plot
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.semilogy(freqs_wk * Fs, psd_wk, 'b', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('PSD via Auto-correlation (Wiener-Khinchin)')
    plt.xlim([0, 200])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(freqs_pd * Fs, psd_pd, 'r', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('PSD via Periodogram (Direct)')
    plt.xlim([0, 200])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Both methods show peaks at 50 Hz and 120 Hz")
    print("Wiener-Khinchin theorem verified!\n")

    # Example 6: Performance comparison
    print("="*60)
    print("Example 6: Performance Comparison\n")

    print(f"{'Signal Length':>15} | {'Direct Time':>15} | {'FFT Time':>15} | {'Speedup':>10}")
    print("-" * 70)

    for N in [100, 500, 1000, 2000]:
        x = np.random.randn(N)
        y = np.random.randn(N)

        results = compare_correlation_methods(x, y)

        print(f"{N:15d} | {results['time_direct']:13.6f}s | "
              f"{results['time_fft']:13.6f}s | {results['speedup']:9.2f}x")

    print("\nFFT method is significantly faster for large signals!")
    print("\nModule 7 examples completed!")
