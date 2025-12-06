# Module 2: 1D Convolution

## Introduction

**Convolution** is the fundamental operation in signal processing that relates the input signal, the system's impulse response, and the output signal. It's the mathematical tool that allows us to predict how a system will respond to any input.

## Mathematical Definition

The discrete convolution of two signals $x[n]$ and $h[n]$ is:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$

Or equivalently (by changing variables):

$$y[n] = \sum_{k=-\infty}^{\infty} h[k] \cdot x[n-k]$$

The $*$ symbol denotes convolution (not multiplication).

## Intuitive Understanding

Think of convolution as **sliding**, **flipping**, and **summing**:

1. **Flip** the impulse response $h[k]$ to get $h[-k]$
2. **Slide** it to position $n$ to get $h[n-k]$
3. **Multiply** element-wise with $x[k]$
4. **Sum** all products to get $y[n]$

### Visual Example

For a simple case:

$$x[n] = [1, 2, 3], \quad h[n] = [0.5, 0.5]$$

The convolution process:

- At $n=0$: $y[0] = 1 \times 0.5 = 0.5$
- At $n=1$: $y[1] = 1 \times 0.5 + 2 \times 0.5 = 1.5$
- At $n=2$: $y[2] = 2 \times 0.5 + 3 \times 0.5 = 2.5$
- At $n=3$: $y[3] = 3 \times 0.5 = 1.5$

Result: $y[n] = [0.5, 1.5, 2.5, 1.5]$

## Properties of Convolution

### 1. Commutativity

$$x[n] * h[n] = h[n] * x[n]$$

### 2. Associativity

$$x[n] * (h_1[n] * h_2[n]) = (x[n] * h_1[n]) * h_2[n]$$

### 3. Distributivity

$$x[n] * (h_1[n] + h_2[n]) = x[n] * h_1[n] + x[n] * h_2[n]$$

### 4. Identity

$$x[n] * \delta[n] = x[n]$$

The impulse is the identity element for convolution.

### 5. Shift Property

$$x[n] * \delta[n - n_0] = x[n - n_0]$$

Convolving with a shifted impulse shifts the signal.

## Output Length

For finite-length signals:

- If $x[n]$ has length $M$
- If $h[n]$ has length $N$
- Then $y[n] = x[n] * h[n]$ has length $M + N - 1$

This is important for practical implementations!

## Convolution as Filtering

When we convolve a signal with an impulse response, we're **filtering** the signal:

### Low-Pass Filter (Smoothing)

$$h[n] = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right] \quad \text{(Moving average)}$$

Smooths the signal by averaging neighboring points.

### High-Pass Filter (Edge Detection)

$$h[n] = [1, -1] \quad \text{(First difference)}$$

Highlights rapid changes in the signal.

## Implementation Methods

### 1. Direct Convolution

Compute the sum directly from the definition.

- Simple to understand
- Computational complexity: $O(M \times N)$

### 2. FFT-Based Convolution

Use the convolution theorem: convolution in time = multiplication in frequency.

- Much faster for long signals
- Computational complexity: $O((M+N) \log(M+N))$

## Circular vs. Linear Convolution

### Linear Convolution

The standard convolution we've described.

### Circular Convolution

Used in FFT-based methods. Assumes signals are periodic.

To get linear convolution from circular:

1. Zero-pad both signals to length $M + N - 1$
2. Compute circular convolution (using FFT)
3. Result is equivalent to linear convolution

## Practical Applications

1. **Audio Processing**: Echo, reverb effects
2. **Communication Systems**: Channel modeling
3. **Image Blurring**: Extend to 2D (next modules)
4. **Financial Analysis**: Moving averages, smoothing
5. **Machine Learning**: Convolutional neural networks (CNNs)

## Example: Noise Reduction

Given a noisy signal, convolve with a moving average filter:

```python
noisy_signal = true_signal + noise
h = [1/5, 1/5, 1/5, 1/5, 1/5]
smoothed = convolve(noisy_signal, h)
```

The output will be smoother but slightly delayed and blurred.

## Next Steps

In Module 3, we'll explore digital signals in depth, including the Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT), which provide an alternative view of filtering.

## Exercises

1. Implement convolution from scratch without using library functions
2. Verify the commutative property experimentally
3. Design a filter to smooth a noisy signal
4. Compare direct vs. FFT-based convolution speeds
5. Explore the effect of different filter lengths on smoothing

See `convolution.py` and `convolution.ipynb` for implementations.
