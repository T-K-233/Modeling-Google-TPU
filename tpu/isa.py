from typing import Any

import torch

from .instruction import instr
from .arch_state import ArchState


# === Address Loading Instructions ===

@instr("inlined_call_operand.hbm")
def inlined_call_operand_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load HBM base address (in granule index, 16-byte units) for an inlined call operand.

    The parser populates args with the byte address; we store byte_addr // 16
    since subsequent sshll.u32 by 4 expects granule index.
    """
    byte_addr, = params
    state.write_xreg(dest_reg, int(byte_addr))


@instr("inlined_call_operand.vmem")
def inlined_call_operand_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load VMEM base address for an inlined call operand."""
    byte_addr, = params
    state.write_xreg(dest_reg, int(byte_addr))


@instr("inlined_call_operand.<no memory space>")
def inlined_call_operand_smem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load SMEM base address (in granule index, 16-byte units) for an inlined call operand."""
    byte_addr, = params
    state.write_xreg(dest_reg, int(byte_addr))


@instr("int_to_ptr.hbm")
def int_to_ptr_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_addr_reg, = params
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(dest_reg, addr)


@instr("int_to_ptr.vmem")
def int_to_ptr_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_addr_reg, = params
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(dest_reg, addr)


# === SFlag Instructions ===

@instr("vsyncpa")
def vsyncpa(state: ArchState, _: str, params: dict[str, Any]):
    addr, value = params
    addr = int(addr, 16) if addr.startswith("0x") else int(addr)
    state.write_sflag(addr, int(value))


@instr("vsyncadd")
def vsyncadd(state: ArchState, _: str, params: dict[str, Any]):
    addr, value = params
    addr = int(addr, 16) if addr.startswith("0x") else int(addr)
    flag_value = state.read_sflag(addr)
    flag_value = (flag_value + int(value)) % 256
    state.write_sflag(addr, flag_value)


# === DMA Transfer (HBM <-> VMEM) Instructions ===

@instr("dma.hbm_to_vmem")
def dma_hbm_to_vmem(state: ArchState, _: str, params: dict[str, Any]):
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = params
    state.write_sflag(int(sync_flag), 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = int(size_in_granules) << 5
    state.write_vmem(dest_addr, state.read_hbm(src_addr, size))


@instr("dma.vmem_to_hbm")
def dma_vmem_to_hbm(state: ArchState, _: str, params: dict[str, Any]):
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = params
    state.write_sflag(int(sync_flag), 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = int(size_in_granules) << 5
    state.write_hbm(dest_addr, state.read_vmem(src_addr, size))


@instr("dma.done.wait")
def dma_done_wait(state: ArchState, dest_reg: str, params: dict[str, Any]):
    # TODO: this does not stall the execution right now
    # it simply clears the sync flag
    # need to implement this eventually
    sync_flag, size_in_granules = params
    sflag_value = state.read_sflag(int(sync_flag))
    # print(f"  SFlag value at address {sync_flag}: {sflag_value}")


# === Scalar Memory Load/Store Instructions ===

@instr("smov")
def smov(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """ Move the base address of a memory region to a scalar register. """
    address, = params
    address = int(address, 16) if address.startswith("0x") else int(address)
    state.write_xreg(dest_reg, address)


# === SALU Instructions ===

@instr("sshll.u32")
def sshll_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_reg, imm = params
    state.write_xreg(dest_reg, state.read_xreg(src_reg) << int(imm))


# === Tensor Memory Load/Store Instructions ===

@instr("vstv")
def vstv(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """ Load and broadcast a scalar value from SMEM into a vector register. """
    src_reg, = params
    address = state.read_xreg(src_reg)
    scalar_data = state.read_smem(address, 4, dtype=torch.float32)
    data = torch.tensor([scalar_data], dtype=torch.float32).repeat(state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, data)


@instr("vld")
def vld(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """ Vector load from VMEM into a vector register. """
    sreg_or_imm, sublane_mask = params

    if "+" in sreg_or_imm:
        reg, offset = sreg_or_imm.split("+", 1)
        reg_val = state.read_xreg(reg.strip())
        offset = offset.strip()
        offset_val = int(offset, 16) if offset.startswith("0x") else int(offset)
        offset_val = offset_val << 2  # TODO: not sure why need to multiply by 4
        address = reg_val + offset_val

    elif sreg_or_imm.startswith("s"):
        reg_val = state.read_xreg(sreg_or_imm)
        address = reg_val

    else:
        offset_val = int(sreg_or_imm, 16) if sreg_or_imm.startswith("0x") else int(sreg_or_imm)
        address = offset_val

    data = state.read_vmem(address, state.vreg_size).reshape(state.num_sublanes, -1)

    if sublane_mask != "255":
        sublane_mask = int(sublane_mask)
        # 8-bit mask: bit i = 1 keep row i, bit i = 0 clear row i to zero
        row_mask = torch.tensor(
            [(sublane_mask >> i) & 1 for i in range(state.num_sublanes)],
            dtype=torch.bool,
            device=data.device,
        ).unsqueeze(1)
        data = data * row_mask

        if state.verbose:
            print(f"\033[90m  Load with mask '{sublane_mask}' -> {row_mask.flatten().int().tolist()}\033[0m")

    state.write_vreg(dest_reg, data)


@instr("vst")
def vst(state: ArchState, _: str, params: dict[str, Any]):
    """ Vector store from a vector register to VMEM. """
    address, sublane_mask, vsrc_reg = params  # optional middle arg: mask (sm:$0xN)
    data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    address = int(address, 16) if address.startswith("0x") else int(address)

    if sublane_mask != "255":
        mask_val = int(sublane_mask)
        if mask_val == 0:
            if state.verbose:
                print("\033[90m  Store with mask '0' -> [] (no rows stored)\033[0m")
            return

        # 8-bit mask: bit i = 1 keep row i, bit i = 0 skip row i.
        row_mask = torch.tensor(
            [(mask_val >> i) & 1 for i in range(state.num_sublanes)],
            dtype=torch.bool,
            device=data.device,
        ).unsqueeze(1)

        # With the current assumption, mask bits are contiguous from lane 0.
        highest_set_bit = mask_val.bit_length() - 1
        rows_to_store = min(state.num_sublanes, highest_set_bit + 1)
        data = data[:rows_to_store, :]

        if state.verbose:
            print(
                f"\033[90m  Store with mask '{mask_val}' -> "
                f"{row_mask.flatten().int().tolist()} (rows [0:{rows_to_store - 1}] stored)\033[0m"
            )

    state.write_vmem(address, data.flatten())


@instr("vst.msk")
def vst_msk(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """ Vector store from a vector register to VMEM with sublane masking. """
    address, sublane_mask, vsrc_reg = params
    data = state.read_vreg(vsrc_reg)
    address = int(address, 16) if address.startswith("0x") else int(address)

    if sublane_mask != "255":
        mask_val = int(sublane_mask)
        if mask_val == 0:
            if state.verbose:
                print("\033[90m  Store with mask '0' -> [] (no rows stored)\033[0m")
            return

        # 8-bit mask: bit i = 1 keep row i, bit i = 0 skip row i.
        row_mask = torch.tensor(
            [(mask_val >> i) & 1 for i in range(state.num_sublanes)],
            dtype=torch.bool,
            device=data.device,
        ).unsqueeze(1)

        # With the current assumption, mask bits are contiguous from lane 0.
        highest_set_bit = mask_val.bit_length() - 1
        rows_to_store = min(state.num_sublanes, highest_set_bit + 1)
        data = data[:rows_to_store, :]

        if state.verbose:
            print(
                f"\033[90m  Store with mask '{mask_val}' -> "
                f"{row_mask.flatten().int().tolist()} (rows [0:{rows_to_store - 1}] stored)\033[0m"
            )

    state.write_vmem(address, data)


# === VPU Instructions ===

@instr("vadd.f32")
def vadd_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    vsrc1_reg, vsrc2_reg = params
    if vsrc1_reg.startswith("v"):
        vsrc1_data = state.read_vreg(vsrc1_reg, dtype=torch.float32)
    else:
        vsrc1_data = float(vsrc1_reg)
    if vsrc2_reg.startswith("v"):
        vsrc2_data = state.read_vreg(vsrc2_reg, dtype=torch.float32)
    else:
        vsrc2_data = float(vsrc2_reg)

    result = vsrc1_data + vsrc2_data
    state.write_vreg(dest_reg, result)


@instr("vrot.slane")
def vrot_slane(state: ArchState, dest_reg: str, params: dict[str, Any]):
    vsrc_reg, shift_amount = params
    shift_amount = int(shift_amount)
    assert shift_amount >= 0 and shift_amount < state.num_sublanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    vdest_data = vsrc_data.roll(shift_amount, dims=0)
    state.write_vreg(dest_reg, vdest_data)
    # breakpoint()


# === Tensor Packing/Unpacking Instructions ===

@instr("vpack.c.bf16")
def vpack_c_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    reg1_or_imm, vsrc2_reg = params

    if reg1_or_imm.startswith("v"):
        vsrc1_data = state.read_vreg(reg1_or_imm, dtype=torch.float32).to(torch.bfloat16)
    else:
        vsrc1_data = torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.bfloat16) + float(reg1_or_imm)
    vsrc2_data = state.read_vreg(vsrc2_reg, dtype=torch.float32).to(torch.bfloat16)
    packed_data = torch.cat([vsrc1_data.unsqueeze(-1), vsrc2_data.unsqueeze(-1)], dim=-1)
    state.write_vreg(dest_reg, packed_data)


@instr("vunpack.c.l.bf16")
def vunpack_c_l_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    vsrc_reg, = params
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    unpacked_data = vsrc_data.to(torch.float32)[:, 0:state.num_lanes]
    state.write_vreg(dest_reg, unpacked_data)


@instr("vunpack.c.h.bf16")
def vunpack_c_h_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    vsrc_reg, = params
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    unpacked_data = vsrc_data.to(torch.float32)[:, state.num_lanes:]
    state.write_vreg(dest_reg, unpacked_data)


@instr("vunpack.i.l.bf16")
def vunpack_i_l_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    vsrc_reg, = params
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    unpacked_data = vsrc_data.to(torch.float32)[:, 0:state.num_lanes]
    state.write_vreg(dest_reg, unpacked_data)


# === XLU Instructions ===


@instr("vxpose.xlu0.b32.start.end")
def vxpose_xlu0_b32_start_end(state: ArchState, _: str, params: dict[str, Any]):
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[:, 0:num_lanes] = vsrc_data


@instr("vxpose.xlu0.b32.start")
def vxpose_xlu0_b32_start(state: ArchState, _: str, params: dict[str, Any]):
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[:, 0:num_lanes] = vsrc_data


@instr("vxpose.xlu0.b32.end")
def vxpose_xlu0_b32_end(state: ArchState, _: str, params: dict[str, Any]):
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    # roll rightwards for a full tile
    state.xlu_buffer = state.xlu_buffer.roll(state.num_lanes, dims=1)
    state.xlu_buffer[:, 0:num_lanes] = vsrc_data


@instr("vpop.trf.xlu0")
def vpop_trf_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """
    Pop the left 8x8 square from the XLU buffer, transpose it, and load it to the destination register.
    All other elements in the destination register are set to zero.

    The XLU buffer is a 16 x 128 array. Tensor registers are first transposed and then
    shifted into the buffer from left to right (increasing column index). The result
    is always read out from the bottom-left 8x8 square, effectively sliding this
    window downward over time.

    XLU buffer (16 x 128):

            cols 0-7             cols 8-255
            +--------+------------------------------------------+
        --> |XXXXXXXX|                                          |
        --> |XXXXXXXX|             XLU Buffer                   |
        --> |XXXXXXXX|                                          |
        --> |XXXXXXXX|                                          |
            +--------+------------------------------------------+
             | | | |
             V V V V
         8x8 submatrix (bottom-left; popped, transposed, and written to dest)
    """
    result = torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32)
    result[:, 0:state.num_sublanes] = state.xlu_buffer[:, 0:state.num_sublanes].clone().transpose(0, 1)
    state.xlu_buffer = state.xlu_buffer.roll(-state.num_sublanes, dims=1)
    state.xlu_buffer[:, -state.num_sublanes:] = 0
    state.write_vreg(dest_reg, result)


# === MXU Instructions ===

@instr("vmatpush.msra.mxu0")
def vmatpush_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu1")
def vmatpush_msra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu2")
def vmatpush_msra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu3")
def vmatpush_msra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu0")
def vmatpush_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.bf16.xpose.msra.mxu0")
def vmatpush_bf16_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.bfloat16))


@instr("vmatmul.f32.gmra.mxu0")
def vmatmul_f32_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.gmra.mxu1")
def vmatmul_f32_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.gmra.mxu2")
def vmatmul_f32_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.gmra.mxu3")
def vmatmul_f32_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu0")
def vmatmul_f32_vlgmr_msra_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu1")
def vmatmul_f32_vlgmr_msra_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu2")
def vmatmul_f32_vlgmr_msra_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu3")
def vmatmul_f32_vlgmr_msra_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vpop.f32.mrf.mxu0")
def vpop_f32_mrf_mxu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    result = state.pop_mxu_accumulator("mxu0")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu1")
def vpop_f32_mrf_mxu1(state: ArchState, dest_reg: str, params: dict[str, Any]):
    result = state.pop_mxu_accumulator("mxu1")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu2")
def vpop_f32_mrf_mxu2(state: ArchState, dest_reg: str, params: dict[str, Any]):
    result = state.pop_mxu_accumulator("mxu2")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu3")
def vpop_f32_mrf_mxu3(state: ArchState, dest_reg: str, params: dict[str, Any]):
    result = state.pop_mxu_accumulator("mxu3")
    state.write_vreg(dest_reg, result)
