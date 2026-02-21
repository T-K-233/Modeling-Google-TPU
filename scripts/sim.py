from pathlib import Path

import torch

from instruction import IsaSpec
from parser import BundleParser
from arch_state import ArchState
import isa  # noqa: F401


class Simulator:
    def __init__(self):
        self.state = ArchState()
        self.cycles = 0
        self.max_cycles = 1000000

    def load_program(self, program_path: str):
        parser = BundleParser()
        self.symbol_table, self.program = parser.parse_program(program_path)

        print("Symbol table:")
        for name, value in self.symbol_table.items():
            print(f"  {name}: {value}")

    def load_program_data(self, symbol_values: dict[str, torch.Tensor]):
        for symbol, value in symbol_values.items():
            value_size = value.numel() * value.itemsize
            assert symbol in self.symbol_table, f"Symbol {symbol} not found in symbol table"
            assert value_size <= self.symbol_table[symbol].size, f"Symbol {symbol} data size exceeds memory size: {value_size} > {self.symbol_table[symbol].size}"
            match self.symbol_table[symbol].space:
                case "hbm":
                    self.state.write_hbm(self.symbol_table[symbol].base_address, value)
                case "smem":  # scalar
                    self.state.write_smem(self.symbol_table[symbol].base_address, value)
                case _:
                    raise ValueError(f"Unknown memory space: {self.symbol_table[symbol].space}")

    def run(self):
        while True:
            if self.state.pc > max(self.program.keys()):
                print(f"Reached end of program at PC={hex(self.state.pc)}")
                break

            if self.cycles >= self.max_cycles:
                print(f"Reached max cycles: {self.cycles}")
                break

            bundle = self.program[self.state.pc]
            for uop in bundle:
                opcode = uop.opcode

                # if opcode in breakpoints:
                #     breakpoint()

                if opcode not in IsaSpec.operations:
                    print(f"WARNING: Unknown opcode: {opcode} {uop.dest_reg} {uop.args}, skipping instruction")
                    continue

                print(f">>> {opcode} \033[94m{uop.dest_reg if uop.dest_reg else '_'}, {uop.args}\033[0m")
                IsaSpec.operations[opcode].apply_effect(self.state, uop.dest_reg, uop.args)

            self.state.next_pc = self.state.pc + 1
            self.state.pc = self.state.next_pc
            self.cycles += 1
