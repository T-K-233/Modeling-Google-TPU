import torch

from tpu.sim import Simulator


if __name__ == "__main__":
    sim = Simulator(verbose=True)

    # Run the full multi-kernel linear_f32 program from tests.
    sim.load_program("./tests/linear_f32/tpu_compiler_dump/llo")

    # Inputs and golden computation mirror tests/linear_f32/source.py.
    x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    w = torch.arange(8 * 128, dtype=torch.float32).reshape(128, 8)
    b = torch.arange(8, dtype=torch.float32)
    golden = x @ w + b

    sim.run_all_kernels()

    assert sim.final_output_address is not None
    result = sim.state.read_hbm(
        sim.final_output_address,
        8 * 8 * torch.float32.itemsize,
        dtype=torch.float32,
    ).reshape(8, 8)

    print("HBM result (first 4x4):")
    print(result[0:4, 0:4])

    print("Golden result (first 4x4):")
    print(golden[0:4, 0:4])

    print("Allclose:", torch.allclose(result, golden, rtol=1e-5, atol=1e-8))
