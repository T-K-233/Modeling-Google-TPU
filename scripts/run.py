from pathlib import Path

import torch

from sim import Simulator


# program_path = Path("./tpu_compiler_dump/add1/llo/1771615577204836121-broadcast_add_fusion")
# program_path = Path("./tpu_compiler_dump/simple/llo/1771630997835859261-add.3")
program_path = Path("./tpu_compiler_dump/test/llo/1771638841886207694-broadcast_add_fusion")


if __name__ == "__main__":
    sim = Simulator()

    sim.load_program(program_path)
    sim.load_program_data({
        "#operand0": torch.ones(8, 128, dtype=torch.float32),
        "#operand1": torch.ones(1, dtype=torch.float32) * 10,
    })
    sim.run()

    # print(sim.state.read_vreg("v2", dtype=torch.float32))
    # print(sim.state.read_vreg("v3", dtype=torch.float32))

    result = sim.state.read_hbm(sim.symbol_table["#operand2"].base_address, 8 * 128 * 4, dtype=torch.float32)
    result = result.reshape(8, 128)
    print(result)

    # assert torch.allclose(result, torch.ones(8, 128, dtype=torch.float32)*2)
