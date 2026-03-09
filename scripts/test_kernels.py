import unittest

import torch

from tpu.sim import Simulator
from tpu.tiling import pack_bf16_register, unpack_bf16_register


class TpuCompleteDumpTests(unittest.TestCase):
    VERBOSE = False

    FP32_RTOL = 1e-5
    FP32_ATOL = 1e-8
    BF16_RTOL = 1.6e-2
    BF16_ATOL = 1e-5

    def setUp(self):
        self.sim = Simulator(verbose=self.VERBOSE)

    def _check_result(self, result: torch.Tensor, golden: torch.Tensor, rtol: float, atol: float):
        if not torch.allclose(result.flatten(), golden.flatten(), rtol=rtol, atol=atol):
            diff = (result - golden).abs()
            max_diff = diff.max().item()
            rel_diff = (diff / golden.abs().clamp(min=1e-12)).max().item()
            self.fail(
                "\n".join(
                    [
                        "Result mismatch",
                        f"  max_abs_diff={max_diff:.6e}",
                        f"  max_rel_diff={rel_diff:.6e}",
                        f"  mismatched={(diff > atol + rtol * golden.abs()).sum().item()}/{result.numel()}",
                    ]
                )
            )

    def _pack_bf16_8x256(self, logical: torch.Tensor) -> torch.Tensor:
        low = logical[:, : self.sim.state.num_lanes]
        high = logical[:, self.sim.state.num_lanes :]
        return pack_bf16_register(low, high, self.sim.state.num_sublanes, self.sim.state.num_lanes)

    def _unpack_bf16_8x256(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(self.sim.state.num_sublanes, self.sim.state.num_lanes * 2)
        low, high = unpack_bf16_register(
            packed, self.sim.state.num_sublanes, self.sim.state.num_lanes
        )
        return torch.cat([low, high], dim=1)

    @staticmethod
    def _source_vector_add_bf16_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors tests/vector_add_bf16/source.py
        a = torch.arange(8 * 128 * 2, dtype=torch.bfloat16).reshape(8, 256)
        b = torch.arange(8 * 128 * 2, dtype=torch.bfloat16).reshape(8, 256)
        return a, b

    @staticmethod
    def _source_vector_add_f32_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors tests/vector_add_f32/source.py
        a = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        b = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        return a, b

    @staticmethod
    def _source_linear_f32_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mirrors tests/linear_f32/source.py
        x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        w = torch.arange(8 * 128, dtype=torch.float32).reshape(128, 8)
        b = torch.arange(8, dtype=torch.float32)
        return x, w, b

    def testVectorAddBf16CompleteDump(self):
        self.sim.load_program("./tests/vector_add_bf16/tpu_compiler_dump/llo")

        operand_b = torch.ones((8, 256), dtype=torch.bfloat16)
        packed_operand_b = self._pack_bf16_8x256(operand_b)
        self.sim.load_program_data({"#param0": packed_operand_b})
        self.sim.run_all_kernels()
        self.assertEqual(self.sim.kernel_call_trace, ["iota.1", "reshape.2", "add.3"])

        assert self.sim.final_output_address is not None
        result = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 256 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        ).reshape(8, 256)

        # Dataflow validation for the complete dump:
        # stage2(add.3) output == stage1(reshape.2) output + external operand.
        stage1_output_addr = self.sim._produced_values[1].address
        stage1 = self.sim.state.read_hbm(
            stage1_output_addr,
            8 * 256 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        ).reshape(8, 256)
        golden = stage1 + packed_operand_b
        self._check_result(result, golden, self.BF16_RTOL, self.BF16_ATOL)

    def testVectorAddF32CompleteDump(self):
        self.sim.load_program("./tests/vector_add_f32/tpu_compiler_dump/llo")

        operand_b = torch.ones((8, 128), dtype=torch.float32)
        self.sim.load_program_data({"#param0": operand_b})
        self.sim.run_all_kernels()
        self.assertEqual(self.sim.kernel_call_trace, ["iota.1", "copy.1", "add.3"])

        assert self.sim.final_output_address is not None
        result = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)

        # Dataflow: add.3 output == copy.1 output + external operand.
        stage1_output_addr = self.sim._produced_values[1].address
        stage1 = self.sim.state.read_hbm(
            stage1_output_addr,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)
        golden = stage1 + operand_b
        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testLinearF32CompleteDump(self):
        self.sim.load_program("./tests/linear_f32/tpu_compiler_dump/llo")
        self.sim.run_all_kernels()
        self.assertEqual(
            self.sim.kernel_call_trace,
            ["iota.1", "copy.1", "reshape.2", "copy", "iota.1", "convolution_add_fusion"],
        )

        assert self.sim.final_output_address is not None
        result = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 8)

        x, w, b = self._source_linear_f32_inputs()
        golden = x @ w + b
        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorAddBf16GoldenFromSource(self):
        # Source: tests/vector_add_bf16/source.py
        x, y = self._source_vector_add_bf16_inputs()
        golden = (x + y).to(torch.bfloat16)

        self.sim.load_program("./tests/vector_add_bf16/tpu_compiler_dump/llo")
        packed_y = self._pack_bf16_8x256(y)
        self.sim.load_program_data({"#param0": packed_y})
        self.sim.run_all_kernels()

        assert self.sim.final_output_address is not None
        result_packed = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 256 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        ).reshape(8, 256)
        result = self._unpack_bf16_8x256(result_packed)
        self._check_result(result, golden, self.BF16_RTOL, self.BF16_ATOL)

    def testLinearF32GoldenFromSource(self):
        # Source: tests/linear_f32/source.py
        x, w, b = self._source_linear_f32_inputs()
        golden = x @ w + b

        self.sim.load_program("./tests/linear_f32/tpu_compiler_dump/llo")
        self.sim.run_all_kernels()

        assert self.sim.final_output_address is not None
        result = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 8)
        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorAddF32GoldenFromSource(self):
        # Source: tests/vector_add_f32/source.py
        x, y = self._source_vector_add_f32_inputs()
        golden = x + y

        self.sim.load_program("./tests/vector_add_f32/tpu_compiler_dump/llo")
        self.sim.load_program_data({"#param0": y})
        self.sim.run_all_kernels()

        assert self.sim.final_output_address is not None
        result = self.sim.state.read_hbm(
            self.sim.final_output_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)
        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TPU simulator logging",
    )
    args, remaining = parser.parse_known_args()
    TpuCompleteDumpTests.VERBOSE = args.verbose
    unittest.main(argv=[sys.argv[0]] + remaining)
