from pathlib import Path

import torch

from tpu.sim import Simulator


program_path = Path("./tests/lane_reduce_bf16/tpu_compiler_dump/llo/1771959547088034911-reduce")


if __name__ == "__main__":
    sim = Simulator(verbose=True)

    a = torch.arange(0, 16*128, dtype=torch.bfloat16).reshape(16, 128)
    # a = torch.ones(16, 128, dtype=torch.bfloat16)

    # b = torch.arange(0, 8*128, dtype=torch.float32).reshape(128, 8)

    # a = torch.ones(8, 128, dtype=torch.float32)
    # b = torch.ones(128, 8, dtype=torch.float32)

    sim.load_program(program_path)
    sim.load_program_data({
        "#operand0": a,
        # "#operand1": b,
    })

    sim.run()

    result = sim.state.read_hbm(sim.symbol_table["#operand2"].base_address, 8 * 128 * 4, dtype=torch.float32)
    result = result.reshape(8, 128)
    print("HBM result:")
    print(result[0:8, 0:10])

    c = a.sum(dim=1)

    # assert torch.allclose(result, torch.ones(8, 128, dtype=torch.float32)*2)

    print("Golden result:")
    print(c)
