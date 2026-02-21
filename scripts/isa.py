from typing import Any

import torch

from instruction import instr
from arch_state import ArchState


@instr("vsyncpa")
def vsyncpa(state: ArchState, _: str, params: dict[str, Any]):
    addr, value = params
    state.write_sflag(int(addr), int(value))


@instr("vsyncadd")
def vsyncadd(state: ArchState, _: str, params: dict[str, Any]):
    addr, value = params
    flag_value = state.read_sflag(int(addr))
    flag_value = (flag_value + int(value)) % 256
    state.write_sflag(int(addr), flag_value)


@instr("int_to_ptr.hbm")
def int_to_ptr_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_addr_reg, = params
    addr = state.read_xreg(src_addr_reg)
    addr = addr
    state.write_xreg(dest_reg, addr)


@instr("int_to_ptr.vmem")
def int_to_ptr_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_addr_reg, = params
    addr = state.read_xreg(src_addr_reg)
    addr = addr >> 4  # not sure why
    state.write_xreg(dest_reg, addr)


@instr("dma.hbm_to_vmem")
def dma_hbm_to_vmem(state: ArchState, _: str, params: dict[str, Any]):
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = params
    state.write_sflag(int(sync_flag), 1)
    src_addr = state.read_xreg(src_addr_reg)
    dest_addr = state.read_xreg(dest_addr_reg)

    # not sure why size_in_bytes = size_in_granules * 32
    # but this ratio is observed from DMA transfer instructions
    size = int(size_in_granules) * 32
    state.write_vmem(dest_addr, state.read_hbm(src_addr, size))


@instr("dma.vmem_to_hbm")
def dma_vmem_to_hbm(state: ArchState, _: str, params: dict[str, Any]):
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = params
    state.write_sflag(int(sync_flag), 1)
    src_addr = state.read_xreg(src_addr_reg)
    dest_addr = state.read_xreg(dest_addr_reg)

    # not sure why size_in_bytes = size_in_granules * 32
    # but this ratio is observed from DMA transfer instructions
    size = int(size_in_granules) * 32
    state.write_hbm(dest_addr, state.read_vmem(src_addr, size))


@instr("dma.done.wait")
def dma_done_wait(state: ArchState, dest_reg: str, params: dict[str, Any]):
    # TODO: this does not stall the execution right now
    # it simply clears the sync flag
    # need to implement this eventually
    sync_flag, size_in_granules = params
    sflag_value = state.read_sflag(int(sync_flag))
    print(f"  Sflag value at address {sync_flag}: {sflag_value}")
    state.write_sflag(int(sync_flag), 0)


@instr("sshll.u32")
def sshll_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_reg, imm = params
    state.write_xreg(dest_reg, state.read_xreg(src_reg) << int(imm))


@instr("smov")
def smov(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """ Move the base address of a memory region to a scalar register. """
    address, = params
    state.write_xreg(dest_reg, int(address))


@instr("inlined_call_operand.hbm")
def inlined_call_operand_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load HBM base address (in granule index, 16-byte units) for an inlined call operand.

    The parser populates args with the byte address; we store byte_addr // 16
    since subsequent sshll.u32 by 4 expects granule index.
    """
    byte_addr, = params
    granule_idx = int(byte_addr) // 16
    state.write_xreg(dest_reg, granule_idx)


@instr("inlined_call_operand.<no memory space>")
def inlined_call_operand_no_mem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Placeholder for scalar operands (e.g. f32[]) - store value 0; host supplies data."""
    byte_addr, = params
    granule_idx = int(byte_addr) // 16
    state.write_xreg(dest_reg, granule_idx)


@instr("vstv")
def vstv(state: ArchState, dest_reg: str, params: dict[str, Any]):
    src_reg, = params
    address = state.read_xreg(src_reg)
    value = state.read_smem(address, 4, dtype=torch.float32)
    state.write_vreg(dest_reg, torch.tensor([value], dtype=torch.float32).repeat(state.v_register_shape))


@instr("vld")
def vld(state: ArchState, dest_reg: str, params: dict[str, Any]):
    address, = params
    size = state.v_register_shape[0] * state.v_register_shape[1] * torch.float32.itemsize
    data = state.read_vmem(int(address), size)
    state.write_vreg(dest_reg, data.view(torch.float32))


@instr("vst")
def vst(state: ArchState, _: str, params: dict[str, Any]):
    address, vsrc_reg = params
    data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.write_vmem(int(address), data.flatten())


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
