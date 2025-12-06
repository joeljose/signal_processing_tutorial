# Module 1: Signals and Systems

## Introduction

A **signal** is a function that conveys information about the behavior or attributes of a phenomenon. In mathematical terms, a signal is a function of one or more independent variables.

## Types of Signals

### 1. Continuous-Time Signals

Signals defined for every value of time $t$ in a continuous interval.

**Example:**

$$x(t) = A \cos(2\pi f t)$$

Properties:

- Defined for all real values of $t$
- Amplitude can vary smoothly
- Common in analog systems

### 2. Discrete-Time Signals

Signals defined only at discrete points in time, typically denoted as $x[n]$ where $n$ is an integer.

**Example:**

$$x[n] = A \cos(2\pi f n)$$

Properties:

- Defined only at integer values of $n$
- Result from sampling continuous signals
- Foundation of digital signal processing

## Key Signal Types

### Unit Step Function

**Continuous:**

$$u(t) = \begin{cases}
1 & \text{for } t \geq 0 \\
0 & \text{for } t < 0
\end{cases}$$

**Discrete:**

$$u[n] = \begin{cases}
1 & \text{for } n \geq 0 \\
0 & \text{for } n < 0
\end{cases}$$

### Unit Impulse (Delta Function)

**Continuous:** $\delta(t)$

- Infinite at $t = 0$
- Zero elsewhere
- Satisfies: $\int_{-\infty}^{\infty} \delta(t) \, dt = 1$

**Discrete:**

$$\delta[n] = \begin{cases}
1 & \text{for } n = 0 \\
0 & \text{for } n \neq 0
\end{cases}$$

The discrete impulse is simpler and more intuitive than the continuous version.

## Linear Time-Invariant (LTI) Systems

An LTI system is characterized by two properties:

### 1. Linearity

If input $x_1(t)$ produces output $y_1(t)$ and $x_2(t)$ produces $y_2(t)$, then:

$$a \cdot x_1(t) + b \cdot x_2(t) \rightarrow a \cdot y_1(t) + b \cdot y_2(t)$$

where $a$ and $b$ are arbitrary constants.

### 2. Time Invariance

If $x(t)$ produces $y(t)$, then:

$$x(t - t_0) \rightarrow y(t - t_0)$$

The system's behavior doesn't change over time.

## Impulse Response

The **impulse response** $h(t)$ or $h[n]$ is the output of an LTI system when the input is an impulse $\delta(t)$ or $\delta[n]$.

**Why is it important?**

- Completely characterizes an LTI system
- Any output can be computed using the impulse response and convolution
- Foundation for filtering and signal processing

### Example: Simple Moving Average

A 3-point moving average filter has impulse response:

$$h[n] = \begin{cases}
\frac{1}{3} & \text{for } n = 0, 1, 2 \\
0 & \text{otherwise}
\end{cases}$$

Or simply: $h[n] = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]$

This smooths the signal by averaging each point with its neighbors.

## System Properties

### Causality

A system is **causal** if the output at time $t$ depends only on inputs at times $\tau \leq t$.

For discrete systems:

$$h[n] = 0 \quad \text{for } n < 0$$

### Stability

A system is **stable** if bounded inputs produce bounded outputs (BIBO stability).

**Condition for stability:**

- **Discrete:** $\sum_{n=-\infty}^{\infty} |h[n]| < \infty$
- **Continuous:** $\int_{-\infty}^{\infty} |h(t)| \, dt < \infty$

## Next Steps

In Module 2, we'll explore **convolution**, which is the mathematical operation that relates input, output, and impulse response:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$

## Exercises

1. Generate and plot unit step and impulse signals
2. Create a simple exponential signal $x[n] = a^n \cdot u[n]$
3. Verify linearity of a simple system
4. Design a 5-point moving average filter and find its impulse response

See `examples.py` and `signals_and_systems.ipynb` for implementations.
