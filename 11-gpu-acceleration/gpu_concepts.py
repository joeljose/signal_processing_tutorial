"""
Module 11: GPU Acceleration Basics
Conceptual demonstrations and examples of GPU computing principles

Note: Some examples require CuPy (CUDA) to run. Install with:
  pip install cupy-cuda11x  # for CUDA 11.x
  pip install cupy-cuda12x  # for CUDA 12.x

For systems without GPU, examples show the code structure and concepts.
"""

import numpy as np
import time

# Try to import CuPy (graceful failure if not available)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU examples will run")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - showing CPU-only examples")
    print("Install CuPy to enable GPU acceleration")


# ============================================================================
# Basic CPU vs GPU Comparison
# ============================================================================

def demonstrate_cpu_vs_gpu():
    """
    Demonstrate basic CPU vs GPU array operations
    """
    print("\nExample: CPU vs GPU Array Operations")
    print("=" * 50)

    size = 10_000_000  # 10 million elements
    print(f"Array size: {size:,} elements")

    # CPU version
    a_cpu = np.random.rand(size).astype(np.float32)
    b_cpu = np.random.rand(size).astype(np.float32)

    start = time.time()
    c_cpu = a_cpu + b_cpu
    cpu_time = (time.time() - start) * 1000
    print(f"CPU time: {cpu_time:.2f} ms")

    if GPU_AVAILABLE:
        # GPU version
        a_gpu = cp.random.rand(size, dtype=cp.float32)
        b_gpu = cp.random.rand(size, dtype=cp.float32)

        # Warm-up (first run includes overhead)
        _ = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()

        # Timed run
        start = time.time()
        c_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_time = (time.time() - start) * 1000

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

        # Verify results match
        c_cpu_check = cp.asnumpy(c_gpu)
        error = np.max(np.abs(c_cpu - c_cpu_check))
        print(f"Max difference: {error:.10f}")
    else:
        print("GPU not available for comparison")


# ============================================================================
# Memory Transfer Overhead
# ============================================================================

def demonstrate_memory_transfer():
    """
    Show impact of CPU-GPU memory transfers
    """
    print("\nExample: Memory Transfer Overhead")
    print("=" * 50)

    if not GPU_AVAILABLE:
        print("Requires GPU/CuPy")
        return

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    print(f"\n{'Size':<12} {'CPU→GPU (ms)':<15} {'GPU→CPU (ms)':<15} {'Computation (ms)':<20}")
    print("-" * 70)

    for size in sizes:
        # Create CPU array
        data_cpu = np.random.rand(size).astype(np.float32)

        # Time transfer to GPU
        start = time.time()
        data_gpu = cp.asarray(data_cpu)
        cp.cuda.Stream.null.synchronize()
        transfer_to_gpu = (time.time() - start) * 1000

        # Time computation on GPU
        start = time.time()
        result_gpu = data_gpu * 2.0 + 1.0
        cp.cuda.Stream.null.synchronize()
        compute_time = (time.time() - start) * 1000

        # Time transfer back to CPU
        start = time.time()
        result_cpu = cp.asnumpy(result_gpu)
        transfer_to_cpu = (time.time() - start) * 1000

        print(f"{size:<12,} {transfer_to_gpu:>10.3f}      {transfer_to_cpu:>10.3f}      {compute_time:>10.3f}")

    print("\nKey insight: Memory transfer often dominates for small arrays!")


# ============================================================================
# Thread Organization
# ============================================================================

def explain_thread_organization():
    """
    Explain CUDA thread hierarchy conceptually
    """
    print("\nExample: Thread Organization")
    print("=" * 50)

    image_width = 1920
    image_height = 1080

    # Typical block configuration
    block_size_x = 16
    block_size_y = 16
    threads_per_block = block_size_x * block_size_y

    # Calculate grid size
    grid_size_x = (image_width + block_size_x - 1) // block_size_x
    grid_size_y = (image_height + block_size_y - 1) // block_size_y
    total_blocks = grid_size_x * grid_size_y

    total_threads = total_blocks * threads_per_block

    print(f"Image size: {image_width} × {image_height} = {image_width * image_height:,} pixels")
    print(f"\nThread organization:")
    print(f"  Block size: {block_size_x} × {block_size_y} = {threads_per_block} threads/block")
    print(f"  Grid size: {grid_size_x} × {grid_size_y} = {total_blocks} blocks")
    print(f"  Total threads: {total_threads:,}")
    print(f"  Threads per pixel: {total_threads / (image_width * image_height):.2f}")

    # Show how thread computes its global position
    print(f"\nThread → Pixel mapping:")
    print(f"  global_x = blockIdx.x * {block_size_x} + threadIdx.x")
    print(f"  global_y = blockIdx.y * {block_size_y} + threadIdx.y")

    # Example for specific block
    block_x, block_y = 5, 3
    thread_x, thread_y = 8, 12

    global_x = block_x * block_size_x + thread_x
    global_y = block_y * block_size_y + thread_y

    print(f"\n  Example: Block({block_x},{block_y}), Thread({thread_x},{thread_y})")
    print(f"  → Pixel position: ({global_x}, {global_y})")


# ============================================================================
# Arithmetic Intensity
# ============================================================================

def calculate_arithmetic_intensity():
    """
    Calculate arithmetic intensity for different operations
    """
    print("\nExample: Arithmetic Intensity Analysis")
    print("=" * 50)

    operations = [
        ("Element-wise add", 1, 12),  # 1 FLOP, 12 bytes (3 arrays × 4 bytes)
        ("Element-wise multiply-add", 2, 12),  # a*b + c
        ("Dot product (N=1000)", 2000, 8000),  # 2N FLOPs, 2N floats read
        ("Matrix multiply (N×N)", None, None),  # Will calculate
        ("3×3 convolution", 18, 40),  # 9 multiplies + 9 adds, 10 floats read
        ("21×21 convolution", 882, 1764),  # 441 mults + 441 adds, 441 floats
    ]

    print(f"\n{'Operation':<25} {'FLOPs':<12} {'Bytes':<12} {'AI (FLOP/byte)':<15} {'Bottleneck'}")
    print("-" * 85)

    for op_name, flops, bytes_transferred in operations:
        if flops is None:  # Matrix multiply
            N = 1000
            flops = 2 * N**3  # 2N³ operations
            bytes_transferred = 3 * N**2 * 4  # 3 matrices of N² floats
            ai = flops / bytes_transferred
        else:
            ai = flops / bytes_transferred

        bottleneck = "Memory" if ai < 10 else "Compute"

        if flops is None:
            flops_str = f"2N³ (N={N})"
            bytes_str = f"3N² × 4"
        else:
            flops_str = f"{flops}"
            bytes_str = f"{bytes_transferred}"

        print(f"{op_name:<25} {flops_str:<12} {bytes_str:<12} {ai:>10.2f}        {bottleneck}")

    print("\nKey insight:")
    print("  AI < 10: Memory-bound (limited by bandwidth)")
    print("  AI > 10: Compute-bound (benefits from more cores)")


# ============================================================================
# Coalesced vs Uncoalesced Memory Access
# ============================================================================

def demonstrate_memory_patterns():
    """
    Show impact of memory access patterns (conceptual)
    """
    print("\nExample: Memory Access Patterns")
    print("=" * 50)

    if not GPU_AVAILABLE:
        print("Requires GPU/CuPy for timing demonstration")
        print("\nConceptual explanation:")
        print("  Coalesced access: Threads access consecutive memory")
        print("    Thread 0: data[0], Thread 1: data[1], Thread 2: data[2], ...")
        print("    → Single memory transaction (FAST)")
        print("\n  Uncoalesced access: Threads access scattered memory")
        print("    Thread 0: data[0], Thread 1: data[100], Thread 2: data[200], ...")
        print("    → Multiple memory transactions (SLOW)")
        return

    size = 10_000_000

    # Coalesced: Sequential access
    data_gpu = cp.arange(size, dtype=cp.float32)

    # Warm-up
    _ = data_gpu * 2
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    result = data_gpu * 2  # Sequential, coalesced
    cp.cuda.Stream.null.synchronize()
    coalesced_time = (time.time() - start) * 1000

    # Uncoalesced: Strided access (simulated with slice)
    indices = cp.arange(0, size, 100, dtype=cp.int32)  # Strided indices

    start = time.time()
    result = data_gpu[indices] * 2  # Scattered, uncoalesced
    cp.cuda.Stream.null.synchronize()
    uncoalesced_time = (time.time() - start) * 1000

    print(f"Coalesced access time: {coalesced_time:.3f} ms")
    print(f"Uncoalesced access time: {uncoalesced_time:.3f} ms")
    print(f"Slowdown: {uncoalesced_time / coalesced_time:.1f}x")


# ============================================================================
# Parallel Reduction
# ============================================================================

def demonstrate_parallel_reduction():
    """
    Show parallel reduction pattern
    """
    print("\nExample: Parallel Reduction (Sum)")
    print("=" * 50)

    size = 16_777_216  # 16M elements (power of 2)
    data_cpu = np.random.rand(size).astype(np.float32)

    # CPU reduction
    start = time.time()
    sum_cpu = np.sum(data_cpu)
    cpu_time = (time.time() - start) * 1000

    print(f"Array size: {size:,} elements")
    print(f"CPU time: {cpu_time:.2f} ms")
    print(f"Sum (CPU): {sum_cpu:.6f}")

    if GPU_AVAILABLE:
        data_gpu = cp.asarray(data_cpu)

        # Warm-up
        _ = cp.sum(data_gpu)
        cp.cuda.Stream.null.synchronize()

        # GPU reduction
        start = time.time()
        sum_gpu = cp.sum(data_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) * 1000

        print(f"GPU time: {gpu_time:.2f} ms")
        print(f"Sum (GPU): {float(sum_gpu):.6f}")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")

    print("\nParallel reduction algorithm (tree-based):")
    print("  Step 1: [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]")
    print("  Step 2: [a₀+a₁, a₂+a₃, a₄+a₅, a₆+a₇]")
    print("  Step 3: [a₀+a₁+a₂+a₃, a₄+a₅+a₆+a₇]")
    print("  Step 4: [a₀+a₁+...+a₇]")
    print(f"  Complexity: O(log N) = O(log {size}) = {int(np.log2(size))} steps")


# ============================================================================
# Occupancy Calculation
# ============================================================================

def calculate_occupancy():
    """
    Calculate theoretical occupancy for different configurations
    """
    print("\nExample: Occupancy Calculation")
    print("=" * 50)

    # Typical GPU specs (e.g., NVIDIA RTX 3080)
    max_threads_per_sm = 1536
    max_blocks_per_sm = 16
    shared_memory_per_sm = 49152  # bytes
    max_registers_per_sm = 65536

    print("GPU Specs (example: RTX 3080-like):")
    print(f"  Max threads per SM: {max_threads_per_sm}")
    print(f"  Max blocks per SM: {max_blocks_per_sm}")
    print(f"  Shared memory per SM: {shared_memory_per_sm:,} bytes")
    print(f"  Registers per SM: {max_registers_per_sm:,}")

    configurations = [
        (64, 0, 32),    # threads, shared_mem_per_block, registers_per_thread
        (128, 0, 32),
        (256, 0, 32),
        (512, 0, 32),
        (256, 16384, 32),  # With shared memory
        (256, 0, 64),      # More registers
    ]

    print(f"\n{'Threads/Block':<15} {'Shared Mem':<15} {'Regs/Thread':<15} {'Occupancy'}")
    print("-" * 70)

    for threads_per_block, shared_mem, regs_per_thread in configurations:
        # Calculate limits
        blocks_by_threads = max_threads_per_sm // threads_per_block
        blocks_by_count = max_blocks_per_sm

        if shared_mem > 0:
            blocks_by_shared = shared_memory_per_sm // shared_mem
        else:
            blocks_by_shared = max_blocks_per_sm

        regs_per_block = threads_per_block * regs_per_thread
        blocks_by_regs = max_registers_per_sm // regs_per_block

        # Actual blocks per SM
        blocks_per_sm = min(blocks_by_threads, blocks_by_count, blocks_by_shared, blocks_by_regs)

        # Occupancy
        active_threads = blocks_per_sm * threads_per_block
        occupancy = active_threads / max_threads_per_sm * 100

        shared_str = f"{shared_mem:,} B" if shared_mem > 0 else "0"

        print(f"{threads_per_block:<15} {shared_str:<15} {regs_per_thread:<15} {occupancy:>6.1f}%")

    print("\nKey insight: Balance threads/block, shared memory, and registers")
    print("Target: 50-100% occupancy for good latency hiding")


# ============================================================================
# Run All Examples
# ============================================================================

def run_all_examples():
    """
    Run all GPU concept demonstrations
    """
    print("Module 11: GPU Acceleration Basics")
    print("=" * 70)

    if GPU_AVAILABLE:
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU Detected: {gpu_info['name'].decode()}")
        print(f"Compute Capability: {gpu_info['major']}.{gpu_info['minor']}")
        print(f"Total Memory: {gpu_info['totalGlobalMem'] / 1e9:.1f} GB")
    else:
        print("\nNo GPU detected - running conceptual examples only")

    demonstrate_cpu_vs_gpu()
    demonstrate_memory_transfer()
    explain_thread_organization()
    calculate_arithmetic_intensity()
    demonstrate_memory_patterns()
    demonstrate_parallel_reduction()
    calculate_occupancy()

    print("\n" + "=" * 70)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
