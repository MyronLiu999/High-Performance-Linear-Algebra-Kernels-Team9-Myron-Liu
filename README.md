# Matrix and Vector Multiplication Optimization Report

## TeamMember

- Myron Mengyuan Liu
- mengyuan1@uchicago.edu

## Overview

This project implements and benchmarks four C++ numerical kernels:

- row-major matrix-vector multiplication
- column-major matrix-vector multiplication
- naive matrix-matrix multiplication
- matrix-matrix multiplication using transposed B

The test sizes are:

- **Small**: 3000 × 3000 for matrix-vector, 500 × 500 for matrix-matrix
- **Medium**: 4000 × 4000 for matrix-vector, 1000 × 1000 for matrix-matrix
- **Large**: 5000 × 5000 for matrix-vector, 1500 × 1500 for matrix-matrix

---

## 1. Pointers vs. References in C++

A reference is an alias for an existing object, while a pointer stores an address. References are usually safer and cleaner when an object must exist. Pointers are more flexible because they can be null, reassigned, and used with pointer arithmetic.

In numerical algorithms, references are useful for readability when working with guaranteed valid objects. Pointers are more suitable for raw arrays and contiguous memory buffers. In this project, pointers were the better choice because the matrices and vectors were stored as flat arrays, and pointer arithmetic made row and column access efficient.

---

## 2. Row-Major vs. Column-Major Storage and Cache Locality

Storage order affects how data is laid out in memory. In row-major format, elements in the same row are contiguous. In column-major format, elements in the same column are contiguous.

This matters because CPUs perform better when memory is accessed sequentially. In the row-major matrix-vector implementation, each row is read in order, which gives good spatial locality. In the column-major version, performance depends on whether the loop order follows the column-major layout.

For matrix-matrix multiplication, the naive version is slower because one matrix is accessed by columns in row-major storage, which creates a strided access pattern. Using a transposed version of B improves locality because both operands can then be read row by row. In benchmarking, this version was faster, especially for medium and large matrices.

---

## 3. CPU Caches and Locality

CPUs use L1, L2, and L3 caches to reduce the cost of memory access. L1 is the smallest and fastest, while L3 is larger and slower but still much faster than RAM.

Two key ideas are:

- **Spatial locality**: nearby memory locations are accessed close together in time.
- **Temporal locality**: recently used data is likely to be used again soon.

In this project, the row-major matrix-vector implementation benefits from spatial locality by scanning rows contiguously. The vector also benefits from temporal locality because it is reused for each row. In matrix-matrix multiplication, transposing B improves spatial locality by turning column access into row access.

---

## 4. Memory Alignment

Memory alignment means storing data at addresses that match natural hardware boundaries. Good alignment can improve performance, especially for SIMD and vectorized instructions.

In this project, alignment was not the main factor. The larger performance differences came from cache locality and loop ordering rather than alignment itself. Standard allocation was sufficient for correctness and normal performance, but alignment would become more important in a more advanced SIMD-based version.

---

## 5. Compiler Optimizations

Compiler optimizations such as inlining, loop simplification, register allocation, and vectorization can significantly improve performance. They helped both the baseline and optimized implementations, but they were most effective when the memory access pattern was already cache-friendly.

For example, contiguous loops are easier for the compiler to optimize than strided ones. However, aggressive optimization also has drawbacks: it makes debugging harder, can increase code size, and may not help much if the real bottleneck is memory access.

---

## 6. Main Bottlenecks and Profiling

The main bottleneck in the initial implementations was memory access, not arithmetic. For large matrices, cache misses and poor locality were more expensive than the multiply-add operations themselves.

The naive matrix-matrix version was especially slow because matrix B was accessed by columns in row-major layout. Profiling showed that improving memory access patterns was more important than reducing arithmetic operations. That is why transposing B gave the biggest improvement.

---

## 7. Reflection on Teamwork

There is a one person team. I will try to have other members in my further homwork.

---

