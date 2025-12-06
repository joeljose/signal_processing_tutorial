import numpy as np
import matplotlib.pyplot as plt


def unit_step(n):
    """
    Generate unit step function u[n].

    Args:
        n: Array of sample indices

    Returns:
        Array with u[n] = 1 for n >= 0, 0 otherwise
    """
    return np.where(n >= 0, 1.0, 0.0)


def unit_impulse(n):
    """
    Generate unit impulse (delta) function δ[n].

    Args:
        n: Array of sample indices

    Returns:
        Array with δ[n] = 1 for n = 0, 0 otherwise
    """
    return np.where(n == 0, 1.0, 0.0)


def exponential_signal(n, a):
    """
    Generate exponential signal x[n] = a^n * u[n].

    Args:
        n: Array of sample indices
        a: Base of exponential

    Returns:
        Exponential signal
    """
    return (a ** n) * unit_step(n)


def sinusoidal_signal(n, frequency, amplitude=1.0, phase=0.0):
    """
    Generate discrete sinusoidal signal.

    Args:
        n: Array of sample indices
        frequency: Normalized frequency f (cycles/sample), typically 0 to 0.5
                   Note: f = F/Fs where F is continuous freq and Fs is sampling rate
        amplitude: Signal amplitude
        phase: Phase offset in radians

    Returns:
        Sinusoidal signal x[n] = A*cos(2*pi*f*n + phi)

    Note:
        - Normalized frequency f is in cycles per sample
        - Angular frequency omega = 2*pi*f radians/sample
        - Nyquist frequency is f = 0.5 (omega = pi)
        - Frequencies above Nyquist alias to lower frequencies
    """
    return amplitude * np.cos(2 * np.pi * frequency * n + phase)


def sample_continuous_signal(F, Fs, duration, amplitude=1.0, phase=0.0):
    """
    Sample a continuous sinusoidal signal to demonstrate sampling relationship.

    Args:
        F: Continuous frequency in Hz
        Fs: Sampling rate in Hz (samples/second)
        duration: Signal duration in seconds
        amplitude: Signal amplitude
        phase: Phase offset in radians

    Returns:
        t: Continuous time array (for plotting)
        x_t: Continuous signal values
        n: Sample indices
        x_n: Sampled signal values
        f: Normalized frequency (F/Fs)

    Example:
        # Sample a 5 Hz signal at 50 Hz for 1 second
        t, x_t, n, x_n, f = sample_continuous_signal(5, 50, 1.0)
        # f = 5/50 = 0.1 cycles/sample
    """
    # Continuous signal (for visualization)
    t = np.linspace(0, duration, 1000)
    x_t = amplitude * np.cos(2 * np.pi * F * t + phase)

    # Sampled signal
    n = np.arange(0, int(duration * Fs))
    f = F / Fs  # Normalized frequency
    x_n = sinusoidal_signal(n, f, amplitude, phase)

    return t, x_t, n, x_n, f


def moving_average_impulse_response(window_size):
    """
    Generate impulse response for moving average filter.

    Args:
        window_size: Number of points to average

    Returns:
        Impulse response h[n]
    """
    return np.ones(window_size) / window_size


def plot_signal(n, x, title="Signal", xlabel="n", ylabel="x[n]", stem_plot=True):
    """
    Plot a discrete signal.

    Args:
        n: Sample indices
        x: Signal values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        stem_plot: If True, use stem plot; otherwise use line plot
    """
    plt.figure(figsize=(10, 4))
    if stem_plot:
        plt.stem(n, x, basefmt=" ")
    else:
        plt.plot(n, x, 'b-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.tight_layout()


def verify_linearity(system_func, x1, x2, a=2.0, b=3.0):
    """
    Verify if a system is linear.

    A system is linear if: system(a*x1 + b*x2) = a*system(x1) + b*system(x2)

    Args:
        system_func: Function representing the system
        x1, x2: Input signals
        a, b: Scalar coefficients

    Returns:
        Boolean indicating if system is linear
    """
    # Left side: system(a*x1 + b*x2)
    combined_input = a * x1 + b * x2
    lhs = system_func(combined_input)

    # Right side: a*system(x1) + b*system(x2)
    rhs = a * system_func(x1) + b * system_func(x2)

    # Check if approximately equal
    is_linear = np.allclose(lhs, rhs)

    return is_linear, lhs, rhs


# Example systems
def scaling_system(x, gain=2.0):
    """Linear system: multiply by constant."""
    return gain * x


def squaring_system(x):
    """Non-linear system: square each sample."""
    return x ** 2


if __name__ == "__main__":
    # Example 1: Basic signals
    print("Example 1: Generating and plotting basic signals\n")

    n = np.arange(-10, 21)

    # Unit step
    u = unit_step(n)
    plot_signal(n, u, title="Unit Step Function u[n]")
    plt.show()

    # Unit impulse
    delta = unit_impulse(n)
    plot_signal(n, delta, title="Unit Impulse Function δ[n]")
    plt.show()

    # Exponential signal
    x_exp = exponential_signal(n, 0.8)
    plot_signal(n, x_exp, title="Exponential Signal x[n] = 0.8^n * u[n]")
    plt.show()

    # Sinusoidal signal
    x_sin = sinusoidal_signal(n, frequency=0.1, amplitude=2.0)
    plot_signal(n, x_sin, title="Sinusoidal Signal (f=0.1, A=2.0)")
    plt.show()

    # Example 2: Sampling demonstration
    print("\nExample 2: Sampling a Continuous Signal\n")

    # Sample a 5 Hz signal at 50 Hz sampling rate
    F = 5  # Hz
    Fs = 50  # samples/second
    duration = 1.0  # seconds

    t, x_t, n, x_n, f = sample_continuous_signal(F, Fs, duration)

    print(f"Continuous frequency F = {F} Hz")
    print(f"Sampling rate Fs = {Fs} Hz")
    print(f"Normalized frequency f = F/Fs = {f} cycles/sample")
    print(f"Period in samples: N = 1/f = {1/f} samples")
    print(f"Angular frequency: ω = 2πf = {2*np.pi*f:.4f} rad/sample\n")

    plt.figure(figsize=(14, 5))

    # Plot continuous and sampled signals
    plt.subplot(1, 2, 1)
    plt.plot(t, x_t, 'b-', linewidth=1.5, label='Continuous signal', alpha=0.7)
    t_samples = n / Fs  # Convert sample indices to time
    plt.stem(t_samples, x_n, linefmt='r-', markerfmt='ro', basefmt=' ', label='Samples')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Sampling: {F} Hz signal at {Fs} Hz rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot discrete signal with sample indices
    plt.subplot(1, 2, 2)
    plt.stem(n[:50], x_n[:50], basefmt=' ')  # Show first 50 samples
    plt.xlabel('Sample index n')
    plt.ylabel('x[n]')
    plt.title(f'Discrete Signal (f = {f}, first 50 samples)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Example 3: Moving average impulse response
    print("\nExample 3: Moving Average Filter Impulse Response\n")

    h_ma = moving_average_impulse_response(5)
    n_h = np.arange(len(h_ma))
    plot_signal(n_h, h_ma, title="5-Point Moving Average Impulse Response")
    plt.show()
    print(f"Impulse response: {h_ma}")
    print(f"Sum of coefficients: {np.sum(h_ma)}")

    # Example 4: Linearity verification
    print("\nExample 4: Verifying System Linearity\n")

    n_test = np.arange(-5, 6)
    x1 = np.random.randn(len(n_test))
    x2 = np.random.randn(len(n_test))

    # Test linear system
    is_linear, _, _ = verify_linearity(scaling_system, x1, x2)
    print(f"Scaling system is linear: {is_linear}")

    # Test non-linear system
    is_linear, _, _ = verify_linearity(squaring_system, x1, x2)
    print(f"Squaring system is linear: {is_linear}")

    print("\nModule 1 examples completed!")
