# Module 1: Signals and Systems

## Introduction

A **signal** is a function that conveys information about the behavior or attributes of a phenomenon. In mathematical terms, a signal is a function of one or more independent variables.

## Types of Signals

### 1. Continuous-Time Signals

Signals defined for every value of time `t` in a continuous interval.

**Example:** x(t) = A·cos(2πft)

Properties:
- Defined for all real values of t
- Amplitude can vary smoothly
- Common in analog systems

### 2. Discrete-Time Signals

Signals defined only at discrete points in time, typically denoted as x[n] where n is an integer.

**Example:** x[n] = A·cos(2πfn)

Properties:
- Defined only at integer values of n
- Result from sampling continuous signals
- Foundation of digital signal processing

## Key Signal Types

### Unit Step Function

**Continuous:**
```
u(t) = 1 for t ≥ 0
       0 for t < 0
```

**Discrete:**
```
u[n] = 1 for n ≥ 0
       0 for n < 0
```

### Unit Impulse (Delta Function)

**Continuous:** δ(t)
- Infinite at t = 0
- Zero elsewhere
- ∫ δ(t) dt = 1

**Discrete:**
```
δ[n] = 1 for n = 0
       0 for n ≠ 0
```

The discrete impulse is simpler and more intuitive than the continuous version.

## Linear Time-Invariant (LTI) Systems

An LTI system is characterized by two properties:

### 1. Linearity
If input x₁(t) produces output y₁(t) and x₂(t) produces y₂(t), then:
```
a·x₁(t) + b·x₂(t) → a·y₁(t) + b·y₂(t)
```

### 2. Time Invariance
If x(t) produces y(t), then x(t - t₀) produces y(t - t₀)

The system's behavior doesn't change over time.

## Impulse Response

The **impulse response** h(t) or h[n] is the output of an LTI system when the input is an impulse δ(t) or δ[n].

**Why is it important?**
- Completely characterizes an LTI system
- Any output can be computed using the impulse response and convolution
- Foundation for filtering and signal processing

### Example: Simple Moving Average

A 3-point moving average filter has impulse response:
```
h[n] = [1/3, 1/3, 1/3]
```

This smooths the signal by averaging each point with its neighbors.

## System Properties

### Causality
A system is causal if the output at time t depends only on inputs at times ≤ t.

For discrete systems: h[n] = 0 for n < 0

### Stability
A system is stable if bounded inputs produce bounded outputs (BIBO stability).

Condition: ∑|h[n]| < ∞ (discrete) or ∫|h(t)|dt < ∞ (continuous)

## Next Steps

In Module 2, we'll explore **convolution**, which is the mathematical operation that relates input, output, and impulse response:

```
y[n] = x[n] * h[n] = ∑ x[k]·h[n-k]
```

## Exercises

1. Generate and plot unit step and impulse signals
2. Create a simple exponential signal x[n] = aⁿ·u[n]
3. Verify linearity of a simple system
4. Design a 5-point moving average filter and find its impulse response

See `examples.py` and `signals_and_systems.ipynb` for implementations.
