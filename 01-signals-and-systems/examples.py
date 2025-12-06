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
        frequency: Normalized frequency (0 to 0.5 for discrete signals)
        amplitude: Signal amplitude
        phase: Phase offset in radians

    Returns:
        Sinusoidal signal
    """
    return amplitude * np.cos(2 * np.pi * frequency * n + phase)


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

    # Example 2: Moving average impulse response
    print("\nExample 2: Moving Average Filter Impulse Response\n")

    h_ma = moving_average_impulse_response(5)
    n_h = np.arange(len(h_ma))
    plot_signal(n_h, h_ma, title="5-Point Moving Average Impulse Response")
    plt.show()
    print(f"Impulse response: {h_ma}")
    print(f"Sum of coefficients: {np.sum(h_ma)}")

    # Example 3: Linearity verification
    print("\nExample 3: Verifying System Linearity\n")

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
