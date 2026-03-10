from pathlib import Path

import torch

from tpu.sim import Simulator


if __name__ == "__main__":
    sim = Simulator(verbose=True)

    # Load only the convolution_add_fusion kernel so we supply x, w, b directly.
    llo_dir = Path("./tests/linear_f32/tpu_compiler_dump/llo")
    kernel_file = sorted(llo_dir.glob("*-convolution_add_fusion-*-final_bundles.txt"))[0]
    sim.load_kernel(kernel_file)

    # Inputs and golden computation.
    # x = torch.ones(8 * 128, dtype=torch.float32).reshape(8, 128)
    # w = torch.ones(8 * 128, dtype=torch.float32).reshape(128, 8)
    # b = torch.ones(8, dtype=torch.float32)

    x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    w = torch.arange(8 * 128, dtype=torch.float32).reshape(128, 8)
    b = torch.arange(8, dtype=torch.float32)

    golden = x @ w + b

    # Operand 0 (x) is in VMEM; operands 1,2,3 (w, b, output) are in HBM.
    def alloc_hbm(t: torch.Tensor) -> int:
        addr = sim._hbm_heap_ptr
        sim._hbm_heap_ptr += t.numel() * t.element_size()
        sim._write_memory("hbm", addr, t)
        return addr

    addr_x_vmem = 0
    sim._write_memory("vmem", addr_x_vmem, x)
    addr_w = alloc_hbm(w)
    addr_b = alloc_hbm(b)
    addr_y = alloc_hbm(torch.zeros_like(golden))

    call_args = [addr_x_vmem, addr_w, addr_b, addr_y]
    (kernel_record,) = sim.programs.values()
    if not sim._try_fastpath_kernel(kernel_record, call_args):
        raise RuntimeError("Fastpath for convolution_add_fusion did not apply")

    result = sim._read_memory(
        "hbm", addr_y, 8 * 8 * torch.float32.itemsize, dtype=torch.float32
    ).reshape(8, 8)

    print("HBM result (first 4x4):")
    print(result[0:4, 0:4])

    print("Golden result (first 4x4):")
    print(golden[0:4, 0:4])

    print("Allclose:", torch.allclose(result, golden, rtol=1e-5, atol=1e-8))
