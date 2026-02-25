# Google TPU Architecture

## Memory Spaces

### High-Bandwidth Memory (HBM)

Each TPU v5e chip has access to `16 GiB` of off-chip HBM [[ref]](https://docs.cloud.google.com/tpu/docs/v5e). All tensor data are initialized in HBM. XLA compiler denotes HBM as S(0) [[ref]](https://openxla.org/xla/shapes#memory_space_identifiers).

### Vector Memory (VMEM)

The `128 MiB` VMEM on TPU v5e acts as a buffer between vector registers and off-chip HBM [[ref]](https://jax-ml.github.io/scaling-book/tpus/#appendix-b-how-does-a-systolic-array-work). Data must first be transferred into VMEM via DMA before being loaded into vector registers; the same applies in the reverse direction for stores. XLA compiler denotes VMEM as S(1).

### Scalar Memory (SMEM)

SMEM is a much smaller memory region used to store data used by the scalar core. XLA compiler denotes VMEM as S(2).

### Scalar Flags (SFLAG)

It is currently unclear whether SFLAG is implemented as a memory region or a register file. It stores flag information used for DMA synchronization.

### Vector Registers (VREG)

Although they are called *vector* registers, VREGs on TPU v5e actually store 2D matrices. The chip provides 64 VREGs in total, and each register is organized in a fixed `(8, 128)` layout. The `128` is the vector lane dimension, and the `8` is the sublane dimension.

For FP32 data, one VREG holds an `(8, 128)` matrix. Lower-precision formats such as BF16 and FP8 pack more values into the same physical register storage [[ref]](https://openxla.org/xla/tiled_layout#examples_of_tiling_formats).

The distinction between **sublane** and **lane** is performance-critical: operations along the sublane dimension (the 8-element direction, within a column) are typically much cheaper than operations that cross lanes (the 128-element direction, within a row).

Because the VREG layout is a fixed `(8, 128)` shape and cannot be further partitioned, storing smaller matrices can be highly inefficient. In the extreme case, storing a single scalar in one VREG uses only 1 out of 1024 FP32 slots, resulting in a utilization of about **0.1%**.

### Vector Mask (VM)

Vector mask registers store 128-bit lane masks, which are used by masked vector load/store instructions.

### Scalar Registers (SREG)

There are also 64 scalar registers, used to store addresses, control information, and scalar floating-point values.


## TPU Functional Units

### Matrix Execution Unit (MXU)

The MXU is a `128 × 128` weight-stationary systolic array. Before matrix multiplication, weights are loaded (pushed) into the MXU.

### Vector Processing Unit (VPU)

The VPU handles the remaining vector arithmetic operations.

### Cross-Lane Unit (XLU)

The XLU handles more expensive cross-lane operations, including row-wise reductions and matrix transposes.

### Vector Load/Store Unit (VLSU)

The VLSU handles vector loads and stores between VREG and VMEM.

### Direct Memory Access (DMA) Unit

The DMA unit handles memory transfers between VMEM and HBM.

### Scalar Arithmetic and Logical Unit (SALU)

The SALU executes scalar arithmetic and logical operations.


## TPU Instructions

TPU employs a Very Long Instruction Word (VLIW) architecture. Instructions are organized into instruction bundles, where instructions within a bundle are clear of dependencies. Each instruction bundle contains two scalar slots, four vector slots, two matrix slots, one miscellaneous slot, and six immediate numbers \[[ref](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf#Scalar-Computation-Unit), [ref](https://www.youtube.com/watch?v=4bGoGjTRT9U&t=482s)].
