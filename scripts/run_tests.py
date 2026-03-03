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

    VERBOSE = False

    FP32_RTOL = 1e-5
    FP32_ATOL = 1e-8
    BF16_RTOL = 1.6e-2
    BF16_ATOL = 1e-5

    def setUp(self):
        self.sim = Simulator(verbose=self.VERBOSE)

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

    def _pack_bf16_8x256(self, logical: torch.Tensor) -> torch.Tensor:
        """Pack logical BF16 [8,256] as TPU register low/high halves."""
        low = logical[:, :self.sim.state.num_lanes]
        high = logical[:, self.sim.state.num_lanes:]
        return pack_bf16_register(low, high, self.sim.state.num_sublanes, self.sim.state.num_lanes)

    def _unpack_bf16_8x256(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(self.sim.state.num_sublanes, self.sim.state.num_lanes * 2)
        low, high = unpack_bf16_register(packed, self.sim.state.num_sublanes, self.sim.state.num_lanes)
        return torch.cat([low, high], dim=1)

    def _pack_bf16_16x128(self, logical: torch.Tensor) -> torch.Tensor:
        """Pack two BF16 8x128 tiles into one 4096B image for 16x128 kernels."""
        first = convert_to_bf16_tile_layout(logical[:8, :], self.sim.state.num_sublanes, self.sim.state.num_lanes)
        second = convert_to_bf16_tile_layout(logical[8:, :], self.sim.state.num_sublanes, self.sim.state.num_lanes)
        return torch.cat([first, second], dim=0).contiguous()

    def _unpack_bf16_16x128(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(self.sim.state.num_sublanes, self.sim.state.num_lanes * 2)
        first = convert_from_bf16_tile_layout(packed[:4, :], self.sim.state.num_sublanes, self.sim.state.num_lanes)
        second = convert_from_bf16_tile_layout(packed[4:, :], self.sim.state.num_sublanes, self.sim.state.num_lanes)
        return torch.cat([first, second], dim=0)

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
        operand_a = torch.arange(8 * 256, dtype=torch.float32).reshape(8, 256).to(torch.bfloat16)
        operand_b = torch.arange(8 * 256, dtype=torch.float32).reshape(8, 256).to(torch.bfloat16)

        self.sim.load_program(
            "./tests/vector_add_bf16/tpu_compiler_dump/llo/1772532131237750636-add.3"
        )
        self.sim.load_program_data({
            "#operand0": self._pack_bf16_8x256(operand_a),
            "#operand1": self._pack_bf16_8x256(operand_b),
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 256 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        )
        result = self._unpack_bf16_8x256(result)
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

    def testVectorAddF32(self):
        a = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        b = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)

        self.sim.load_program(
            "./tests/vector_add_f32/tpu_compiler_dump/llo/1772532057680416943-add.3"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": b,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)

        self._check_result(result, a + b, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorBroadcastAddF32(self):
        a = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        b = torch.arange(128, dtype=torch.float32)

        self.sim.load_program(
            "./tests/vector_broadcast_add_f32/tpu_compiler_dump/llo/1772532166320759302-broadcast_add_fusion"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": b,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)

        self._check_result(result, a + b, self.FP32_RTOL, self.FP32_ATOL)

    def testReduceRowF32(self):
        a = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        init = torch.tensor(0.0, dtype=torch.float32)

        self.sim.load_program(
            "./tests/reduce_row_f32/tpu_compiler_dump/llo/1772532176587687699-reduce.7"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": init,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).unsqueeze(1)
        golden = a.sum(dim=1, keepdim=True)

        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testReduceColumnF32(self):
        a = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        init = torch.tensor(0.0, dtype=torch.float32)

        self.sim.load_program(
            "./tests/reduce_column_f32/tpu_compiler_dump/llo/1772532184425599243-reduce.7"
        )
        self.sim.load_program_data({
            "#operand0": a,
            "#operand1": init,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(1, 128)
        golden = a.sum(dim=0, keepdim=True)

        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testVectorAddBf16_16x128(self):
        a = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128).to(torch.bfloat16)
        b = torch.ones((16, 128), dtype=torch.bfloat16)

        self.sim.load_program(
            "./tests/vector_add_bf16_16x128/tpu_compiler_dump/llo/1772449856279443160-vadd_16x128"
        )
        self.sim.load_program_data({
            "#operand0": self._pack_bf16_16x128(a),
            "#operand1": self._pack_bf16_16x128(b),
        })
        self.sim.run()

        packed_result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            16 * 128 * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
        )
        result = self._unpack_bf16_16x128(packed_result)
        golden = a + b

        self._check_result(result, golden, self.BF16_RTOL, self.BF16_ATOL)

    def testMatmulF32(self):
        """Matmul kernel lowered from jax matmul_f32 (logical 8x8 operands)."""
        x = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        y = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)

        self.sim.load_program(
            "./tests/matmul_f32/tpu_compiler_dump/llo/1772532192168312817-fusion"
        )
        # Symbol sizes for VMEM/HBM operands are logical (8x8). The VLD path
        # consumes full register images, so write padded physical tiles.
        self.sim.load_program_data({
            "#operand0": torch.zeros((8, 8), dtype=torch.float32),
            "#operand1": torch.zeros((8, 8), dtype=torch.float32),
        })
        x_tile = torch.zeros((8, 128), dtype=torch.float32)
        x_tile[:, :8] = x
        y_tile = torch.zeros((8, 128), dtype=torch.float32)
        y_tile[:, :8] = y
        self.sim.state.write_vmem(self.sim.symbol_table["#operand0"].base_address, x_tile)
        self.sim.state.write_hbm(self.sim.symbol_table["#operand1"].base_address, y_tile)
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand2"].base_address,
            8 * 128 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(8, 128)[:, :8]
        golden = x @ y

        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testLinearTanhF32(self):
        x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        w = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
        b = torch.arange(8, dtype=torch.float32)

        self.sim.load_program(
            "./tests/linear_tanh_f32/tpu_compiler_dump/llo/1772532215965981008-fusion.1"
        )
        self.sim.load_program_data({
            "#operand0": x,
            "#operand1": w,
            "#operand2": b,
        })
        self.sim.run()

        result = self.sim.state.read_hbm(
            self.sim.symbol_table["#operand3"].base_address,
            8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(1, 8)
        golden = torch.tanh(x @ w + b).sum(dim=1).reshape(1, 8)

        self._check_result(result, golden, self.FP32_RTOL, self.FP32_ATOL)

    def testSoftmaxF32ReduceKernel(self):
        """Test softmax pre-reduction kernel: row-wise sum(exp(x - row_max))."""
        x = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        row_max = x.max(dim=1).values

        self.sim.load_program(
            "./tests/softmax_f32/tpu_compiler_dump/llo/1772533135288721407-fusion.1"
        )
        self.sim.load_program_data({
            "#operand0": x,
            "#operand1": row_max,
            "#operand2": torch.zeros(8, dtype=torch.float32),
        })
        self.sim.run()

        result = self.sim.state.read_vmem(
            self.sim.symbol_table["#operand2"].base_address,
            8 * torch.float32.itemsize,
            dtype=torch.float32,
        ).reshape(1, 8)
        golden = torch.exp(x - row_max.unsqueeze(1)).sum(dim=1).reshape(1, 8)

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

    TpuTests.VERBOSE = args.verbose

    unittest.main(argv=[sys.argv[0]] + remaining)
