from pathlib import Path

import torch

from tpu.sim import Simulator


if __name__ == "__main__":
    sim = Simulator(verbose=True)

    # Locate and load only the convolution_add_fusion kernel.
    llo_dir = Path("./tests/linear_f32/tpu_compiler_dump/llo")
    kernel_file = sorted(llo_dir.glob("*-convolution_add_fusion-*-final_bundles.txt"))[0]
    sim.load_kernel(kernel_file)

    # Inputs and golden computation mirror tests/linear_f32/source.py.
    x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    w = torch.arange(8 * 128, dtype=torch.float32).reshape(128, 8)
    b = torch.arange(8, dtype=torch.float32)
    golden = x @ w + b

    # Allocate inputs/outputs in HBM manually.
    def alloc_hbm(t: torch.Tensor) -> int:
        addr = sim._hbm_heap_ptr
        sim._hbm_heap_ptr += t.numel() * t.element_size()
        sim._write_memory("hbm", addr, t)
        return addr

    addr_x = alloc_hbm(x)
    addr_w = alloc_hbm(w)
    addr_b = alloc_hbm(b)
    addr_y = alloc_hbm(torch.zeros_like(golden))

    # Call only the convolution_add_fusion kernel fastpath.
    call_args = [addr_x, addr_w, addr_b, addr_y]
    # There is exactly one loaded kernel program.
    (kernel_record,) = sim.programs.values()
    if not sim._try_fastpath_kernel(kernel_record, call_args):
        raise RuntimeError("Fastpath for convolution_add_fusion did not apply")

    # Read back the kernel output from HBM.
    result = sim._read_memory("hbm", addr_y, 8 * 8 * torch.float32.itemsize, dtype=torch.float32).reshape(8, 8)

    print("Kernel-only HBM result (first 4x4):")
    print(result[0:4, 0:4])

    print("Golden result (first 4x4):")
    print(golden[0:4, 0:4])

    print("Allclose (kernel vs golden):", torch.allclose(result, golden, rtol=1e-5, atol=1e-8))
