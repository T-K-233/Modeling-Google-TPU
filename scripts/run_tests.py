import unittest

import torch

from tpu.sim import Simulator
from tpu.tiling import (
    convert_to_bf16_tile_layout,
    convert_from_bf16_tile_layout,
    pack_bf16_register,
    unpack_bf16_register,
    pack_bf16_mxu_rhs_coalesced,
)


class TpuTests(unittest.TestCase):
    """Test TPU simulator."""

    FP32_RTOL = 1e-5
    FP32_ATOL = 1e-8
    BF16_RTOL = 1.6e-2
    BF16_ATOL = 1e-5

    def setUp(self):
        self.sim = Simulator()

    def _check_result(self, result: torch.Tensor, golden_result: torch.Tensor, rtol: float, atol: float):
        if not torch.allclose(result.flatten(), golden_result.flatten(), rtol=rtol, atol=atol):
            diff = (result - golden_result).abs()
            max_diff = diff.max().item()
            rel_diff = (diff / golden_result.abs().clamp(min=1e-12)).max().item()
            msg = (
                "\033[91m✗ Test failed\033[0m\n"
                f"  Max absolute difference: {max_diff:.6e}\n"
                f"  Max relative difference: {rel_diff:.6e}\n"
                f"  Mismatched elements: {(diff > atol + rtol * golden_result.abs()).sum().item()} / {result.numel()}\n"
                "  Result (first 3×3):\n"
                f"{result[:3, :3]}\n"
                "  Expected (first 3×3):\n"
                f"{golden_result[:3, :3]}"
            )
            self.fail(msg)

    def testVectorAddSimple(self):
        operand_a = torch.ones(8, 128, dtype=torch.float32) * 2
        operand_b = torch.ones(8, 128, dtype=torch.float32)

        self.sim.load_program(
            "./tests/vector_add_simple/tpu_compiler_dump/llo/1771656547132443545-vadd"
        )
        self.sim.load_program_data({
            "#operand0": operand_a,
            "#operand1": operand_b,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        )
        result = result.reshape(8, 128)
        golden_result = operand_a + operand_b

        self._check_result(result, golden_result, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorAdd(self):
        operand_a = torch.ones(16, 256, dtype=torch.float32) * 2
        operand_b = torch.ones(16, 256, dtype=torch.float32)

        self.sim.load_program(
            "./tests/vector_add/tpu_compiler_dump/llo/1771655414213212876-vadd"
        )
        self.sim.load_program_data({
            "#operand0": operand_a,
            "#operand1": operand_b,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            16 * 256 * torch.float32.itemsize,
            dtype=torch.float32,
        )
        result = result.reshape(16, 256)
        golden_result = operand_a + operand_b

        self._check_result(result, golden_result, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorAddBf16(self):
        operand_a = torch.arange(8 * 128, dtype=torch.bfloat16).reshape(8, 128)
        operand_b = torch.arange(8 * 128, dtype=torch.bfloat16).reshape(8, 128)

        self.sim.load_program(
            "./tests/vector_add_bf16/tpu_compiler_dump/llo/1771657700010690791-vadd"
        )
        self.sim.load_program_data({
            "#operand0": convert_to_bf16_tile_layout(operand_a, self.sim.state.num_sublanes, self.sim.state.num_lanes),
            "#operand1": convert_to_bf16_tile_layout(operand_b, self.sim.state.num_sublanes, self.sim.state.num_lanes),
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        )
        result = convert_from_bf16_tile_layout(result, self.sim.state.num_sublanes, self.sim.state.num_lanes)
        golden_result = operand_a + operand_b

        self._check_result(result, golden_result, self.BF16_RTOL, self.BF16_ATOL)

    def testBf16TileLayout(self):
        """Check TPU BF16 row-pair packing: (8,128) -> (4,128,2)."""
        logical = torch.arange(8 * 128, dtype=torch.bfloat16).reshape(8, 128)
        packed = convert_to_bf16_tile_layout(logical, self.sim.state.num_sublanes, self.sim.state.num_lanes)
        packed_3d = packed.reshape(4, 128, 2)
        restored = convert_from_bf16_tile_layout(packed, self.sim.state.num_sublanes, self.sim.state.num_lanes)

        self.assertTrue(torch.equal(packed_3d[:, :, 0], logical[0::2, :]))
        self.assertTrue(torch.equal(packed_3d[:, :, 1], logical[1::2, :]))
        self.assertTrue(torch.equal(restored, logical))

    def testBf16RegisterPackUnpack(self):
        """Check BF16 low/high halves in one packed vector register image."""
        low = torch.arange(8 * 128, dtype=torch.bfloat16).reshape(8, 128)
        high = (torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128) + 10000).to(torch.bfloat16)

        packed = pack_bf16_register(low, high, self.sim.state.num_sublanes, self.sim.state.num_lanes)
        low_out, high_out = unpack_bf16_register(packed, self.sim.state.num_sublanes, self.sim.state.num_lanes)

        self.assertTrue(torch.equal(low_out, low))
        self.assertTrue(torch.equal(high_out, high))

    def testMatmulSimple(self):
        """Matmul C = A @ B with A (8,128), B (128,8) -> C (8,8). Refs scripts/run.py."""
        a = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(8, 128)
        b = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(128, 8)
        b_tiled = b.reshape(16, 8, 8).permute(1, 0, 2).reshape(8, 128).contiguous()

        self.sim.load_program(
            "./tests/matmul_simple/tpu_compiler_dump/llo/1771659663124722556-matmul"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": b_tiled,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        )
        result = result.reshape(8, 128)
        result_8x8 = result[:, 0:8]
        golden_result = a @ b

        self._check_result(result_8x8, golden_result, self.FP32_RTOL, self.FP32_ATOL)

    def testMatmulBf16(self):
        """Matmul C = A @ B with A (8,128), B (128,8) in BF16 -> C (8,8) in F32."""
        a = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(8, 128).to(torch.bfloat16)
        b = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(128, 8).to(torch.bfloat16)

        self.sim.load_program(
            "./tests/matmul_bf16/tpu_compiler_dump/llo/1772449678630211370-matmul_bf16"
        )
        self.sim.load_program_data({
            "#operand0": convert_to_bf16_tile_layout(a, self.sim.state.num_sublanes, self.sim.state.num_lanes),
            # Operand1 follows the MXU RHS coalesced VMEM layout ([8,256] BF16 image).
            "#operand1": pack_bf16_mxu_rhs_coalesced(b),
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        )
        result_8x8 = result.reshape(8, 128)[:, 0:8]
        golden_result = a.float() @ b.float()

        self._check_result(result_8x8, golden_result, self.FP32_RTOL, self.FP32_ATOL)

    def testMatmulBf16RhsTranspose(self):
        """Matmul C = A @ B^T with A/B (8,128) in BF16 -> C (8,8) in F32."""
        a = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(8, 128).to(torch.bfloat16)
        b = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(8, 128).to(torch.bfloat16)

        self.sim.load_program(
            "./tests/matmul_bf16_rhs_transpose/tpu_compiler_dump/llo/1772449809768374815-matmul_bf16_rhs_transpose"
        )
        self.sim.load_program_data({
            "#operand0": convert_to_bf16_tile_layout(a, self.sim.state.num_sublanes, self.sim.state.num_lanes),
            "#operand1": convert_to_bf16_tile_layout(b, self.sim.state.num_sublanes, self.sim.state.num_lanes),
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)[:, 0:8]
        golden_result = a.float() @ b.float().transpose(0, 1)

        self._check_result(result, golden_result, self.FP32_RTOL, self.FP32_ATOL)

    def testLaneReduce(self):
        """Lane reduction: sum over columns of (8,128) input -> (8,). Refs scripts/run.py."""
        a = torch.arange(0, 8 * 128, dtype=torch.float32).reshape(8, 128)
        scalar_init = torch.tensor(0.0, dtype=torch.float32)

        self.sim.load_program(
            "./tests/lane_reduce/tpu_compiler_dump/llo/1771891793285284691-reduce.7"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": scalar_init,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).unsqueeze(1)

        golden_result = a.sum(dim=1, keepdim=True)

        self._check_result(result, golden_result, self.FP32_RTOL, self.FP32_ATOL)


if __name__ == "__main__":
    unittest.main()
