import numpy as np
import matplotlib.pyplot as plt


def dtft(x, omega):
    """
    Compute the Discrete-Time Fourier Transform.

    Args:
        x: Input signal (finite length)
        omega: Array of frequencies at which to evaluate DTFT (radians/sample)

    Returns:
        X: DTFT evaluated at frequencies omega

    Note:
        For finite-length signals, DTFT is:
        X(e^jω) = Σ x[n]e^(-jωn) for n=0 to N-1
    """
    x = np.asarray(x)
    omega = np.asarray(omega)
    n = np.arange(len(x))

    # Compute DTFT using matrix multiplication
    # X(ω) = Σ x[n]e^(-jωn)
    X = np.zeros(len(omega), dtype=complex)
    for i, w in enumerate(omega):
        X[i] = np.sum(x * np.exp(-1j * w * n))

    return X


def dtft_vectorized(x, omega):
    """
    Vectorized DTFT computation (faster for many frequency points).

    Args:
        x: Input signal (finite length)
        omega: Array of frequencies (radians/sample)

    Returns:
        X: DTFT evaluated at frequencies omega
    """
    x = np.asarray(x).reshape(-1, 1)  # Column vector
    omega = np.asarray(omega).reshape(1, -1)  # Row vector
    n = np.arange(len(x)).reshape(-1, 1)  # Column vector

    # X(ω) = Σ x[n]e^(-jωn)
    # Broadcasting: (N,1) * exp(-j * (1,M) * (N,1))
    X = np.sum(x * np.exp(-1j * omega * n), axis=0)

    return X


def plot_dtft(omega, X, title="DTFT"):
    """
    Plot magnitude and phase of DTFT.

    Args:
        omega: Frequency array (radians/sample)
        X: DTFT values (complex)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Magnitude
    axes[0].plot(omega, np.abs(X), 'b', linewidth=2)
    axes[0].set_ylabel('|X(e^{jω})|')
    axes[0].set_title(f'{title} - Magnitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([omega[0], omega[-1]])

    # Phase
    axes[1].plot(omega, np.angle(X), 'r', linewidth=2)
    axes[1].set_xlabel('ω (radians/sample)')
    axes[1].set_ylabel('∠X(e^{jω}) (radians)')
    axes[1].set_title(f'{title} - Phase')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([omega[0], omega[-1]])

    # Mark important frequencies
    for ax in axes:
        ax.axvline(x=-np.pi, color='k', linestyle='--', alpha=0.3, label='-π')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='0')
        ax.axvline(x=np.pi, color='k', linestyle='--', alpha=0.3, label='π')

    axes[0].legend()
    plt.tight_layout()


def demonstrate_aliasing(F_signal, F_s_list, duration=1.0):
    """
    Demonstrate aliasing by sampling a signal at different rates.

    Args:
        F_signal: Frequency of sinusoidal signal (Hz)
        F_s_list: List of sampling rates to try (Hz)
        duration: Signal duration (seconds)
    """
    # Continuous time signal (for reference)
    t_cont = np.linspace(0, duration, 1000)
    x_cont = np.cos(2 * np.pi * F_signal * t_cont)

    num_plots = len(F_s_list) + 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))

    # Plot continuous signal
    axes[0].plot(t_cont, x_cont, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Original Continuous Signal: {F_signal} Hz')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, min(0.1, duration)])

    # Plot sampled signals
    for idx, F_s in enumerate(F_s_list, start=1):
        # Sample the signal
        n_samples = int(duration * F_s)
        t_samples = np.arange(n_samples) / F_s
        x_samples = np.cos(2 * np.pi * F_signal * t_samples)

        # Normalized frequency
        f_norm = F_signal / F_s
        f_alias = f_norm % 1.0  # Fold into [0, 1)
        if f_alias > 0.5:
            f_alias = 1.0 - f_alias
        F_apparent = f_alias * F_s

        # Plot
        axes[idx].plot(t_cont, x_cont, 'b-', linewidth=1, alpha=0.3, label='Original')
        axes[idx].stem(t_samples, x_samples, linefmt='r-', markerfmt='ro',
                       basefmt=' ', label=f'Samples (Fs={F_s} Hz)')

        # Nyquist condition
        nyquist = "✓" if F_s >= 2 * F_signal else "✗"
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(
            f'Fs = {F_s} Hz (f = {f_norm:.3f}, Nyquist {nyquist}) → '
            f'Appears as {F_apparent:.1f} Hz'
        )
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        axes[idx].set_xlim([0, min(0.1, duration)])

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()


def frequency_response_lti(h, omega):
    """
    Compute frequency response of an LTI system.

    Args:
        h: Impulse response
        omega: Frequency array (radians/sample)

    Returns:
        H: Frequency response H(e^jω) = DTFT{h[n]}
    """
    return dtft_vectorized(h, omega)


def plot_frequency_response(h, omega, title="System Frequency Response"):
    """
    Plot magnitude and phase response of an LTI system.

    Args:
        h: Impulse response
        omega: Frequency array
        title: Plot title
    """
    H = frequency_response_lti(h, omega)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Magnitude response (in dB)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    axes[0].plot(omega / np.pi, mag_db, 'b', linewidth=2)
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'{title} - Magnitude Response')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3dB')
    axes[0].legend()

    # Phase response
    axes[1].plot(omega / np.pi, np.angle(H), 'r', linewidth=2)
    axes[1].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title(f'{title} - Phase Response')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])

    plt.tight_layout()


if __name__ == "__main__":
    print("Module 3: DTFT and Sampling Examples\n")

    # Example 1: DTFT of an exponential signal
    print("Example 1: DTFT of Exponential Signal\n")

    a = 0.8
    N = 50
    n = np.arange(N)
    x = a ** n  # x[n] = a^n for n = 0, 1, ..., N-1

    # Compute DTFT
    omega = np.linspace(-np.pi, np.pi, 1000)
    X = dtft_vectorized(x, omega)

    # Analytical solution for comparison: X(e^jω) = (1 - a^N e^(-jωN)) / (1 - a e^(-jω))
    numerator = 1 - a**N * np.exp(-1j * omega * N)
    denominator = 1 - a * np.exp(-1j * omega)
    # Avoid division by zero when denominator is very small
    X_analytical = np.where(np.abs(denominator) > 1e-10,
                            numerator / denominator,
                            N * np.ones_like(omega))  # Limit value when a*e^(-jω) ≈ 1

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(omega, np.abs(X), 'b-', linewidth=2, label='Computed')
    plt.plot(omega, np.abs(X_analytical), 'r--', linewidth=1, label='Analytical')
    plt.xlabel('ω (radians/sample)')
    plt.ylabel('|X(e^{jω})|')
    plt.title(f'DTFT Magnitude: x[n] = {a}^n, n=0...{N-1}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.stem(n[:20], x[:20], basefmt=' ')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.title('Signal (first 20 samples)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 2: Aliasing demonstration
    print("\nExample 2: Aliasing Demonstration\n")

    F_signal = 50  # Signal frequency: 50 Hz
    sampling_rates = [200, 100, 80, 60]  # Try different sampling rates

    print(f"Signal frequency: {F_signal} Hz")
    print(f"Nyquist rate: {2 * F_signal} Hz\n")

    for F_s in sampling_rates:
        nyquist_ok = "✓" if F_s >= 2 * F_signal else "✗"
        f_norm = F_signal / F_s
        print(f"Fs = {F_s:3d} Hz: f = {f_norm:.3f}, Nyquist {nyquist_ok}")

    demonstrate_aliasing(F_signal, sampling_rates, duration=0.5)
    plt.show()

    # Example 3: Frequency response of moving average filter
    print("\nExample 3: Frequency Response of Moving Average Filter\n")

    # 5-point moving average
    M = 5
    h = np.ones(M) / M

    omega = np.linspace(0, np.pi, 1000)
    plot_frequency_response(h, omega, f'{M}-Point Moving Average Filter')
    plt.show()

    print(f"\nImpulse response: {h}")
    print(f"This is a lowpass filter that attenuates high frequencies.")

    # Example 4: Verify periodicity of DTFT
    print("\nExample 4: Periodicity of DTFT\n")

    x = np.array([1, 2, 3, 2, 1])

    omega = np.linspace(-3 * np.pi, 3 * np.pi, 2000)
    X = dtft_vectorized(x, omega)

    plt.figure(figsize=(14, 5))
    plt.plot(omega / np.pi, np.abs(X), 'b', linewidth=2)
    plt.xlabel('ω/π')
    plt.ylabel('|X(e^{jω})|')
    plt.title('DTFT Periodicity: Period = 2π')
    plt.grid(True, alpha=0.3)

    # Mark periods
    for k in range(-1, 2):
        plt.axvline(x=2 * k, color='r', linestyle='--', alpha=0.5)
        plt.text(2 * k, plt.ylim()[1] * 0.9, f'2πk (k={k})',
                 ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    print("The DTFT repeats every 2π radians, as expected!")

    print("\nModule 3 examples completed!")
