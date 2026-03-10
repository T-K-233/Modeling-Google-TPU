from typing import Any

import torch

from .instruction import (
    instr,
    BundleSlotType,
    MXUSlotParams, XLUSlotParams, VALUSlotParams, EUPParams,
    LOADParams, STOREParams, SALUSlotParams,
)
from .arch_state import ArchState
from .tiling import pack_bf16_register, unpack_bf16_register


U32_MASK = 0xFFFFFFFF


def _parse_int(token: Any) -> int:
    if isinstance(token, bool):
        return int(token)
    if isinstance(token, int):
        return token
    if isinstance(token, float):
        return int(token)
    text = str(token).strip()
    if text.startswith("$"):
        text = text[1:]
    if text.startswith("-0x"):
        return -int(text[3:], 16)
    if text.startswith("0x"):
        return int(text, 16)
    return int(text)


def _parse_operand(state: ArchState, token: Any) -> int:
    """Resolve an operand symbol into either a register or an immediate value."""
    if isinstance(token, tuple) and len(token) == 2:
        base_reg, offset = token
        if isinstance(base_reg, str):
            return state.read_xreg(base_reg) + _parse_int(offset)
    if isinstance(token, str) and token.startswith("s"):
        return state.read_xreg(token)
    return _parse_int(token)


def _as_u32(value: int) -> int:
    return value & U32_MASK


def _as_i32(value: int) -> int:
    value &= U32_MASK
    return value - (1 << 32) if value & (1 << 31) else value


def _predicate_active(state: ArchState, params: Any) -> bool:
    pred = params.pred
    if pred is None:
        return True
    pred_token = str(pred)
    if pred_token.startswith("!"):
        return not state.read_preg(pred_token[1:])
    if pred_token.startswith("p"):
        return state.read_preg(pred_token)
    return True


def _is_float_token(token: Any) -> bool:
    if isinstance(token, float):
        return True
    if isinstance(token, int):
        return False
    text = str(token).strip().lower()
    if text in ("nan", "+nan", "-nan", "inf", "+inf", "-inf"):
        return True
    return any(ch in text for ch in (".", "e", "E"))


def _parse_float(token: Any) -> float:
    if isinstance(token, float):
        return token
    if isinstance(token, int):
        return float(token)
    text = str(token).strip().lower()
    if text in ("nan", "+nan", "-nan"):
        return float("nan")
    if text in ("inf", "+inf"):
        return float("inf")
    if text == "-inf":
        return float("-inf")
    return float(text)


def _full_u32(state: ArchState, value: int) -> torch.Tensor:
    return torch.full(
        (state.num_sublanes, state.num_lanes),
        _as_u32(value),
        dtype=torch.uint32,
    )


def _full_f32(state: ArchState, value: float) -> torch.Tensor:
    return torch.full(
        (state.num_sublanes, state.num_lanes),
        float(value),
        dtype=torch.float32,
    )


def _vector_operand_u32(state: ArchState, token: Any) -> torch.Tensor:
    if isinstance(token, str) and (token.startswith("v") or token in state.vreg):
        return state.read_vreg(token, dtype=torch.uint32).clone()
    if _is_float_token(token):
        return _full_f32(state, _parse_float(token)).view(torch.uint32)
    return _full_u32(state, _parse_int(token))


def _vector_operand_i32(state: ArchState, token: Any) -> torch.Tensor:
    return _vector_operand_u32(state, token).view(torch.int32)


def _vector_operand_u16(state: ArchState, token: Any) -> torch.Tensor:
    if isinstance(token, str) and (token.startswith("v") or token in state.vreg):
        return state.read_vreg(token, dtype=torch.uint16).clone()
    return torch.full(
        (state.num_sublanes, state.num_lanes * 2),
        _parse_int(token) & 0xFFFF,
        dtype=torch.uint16,
    )


def _vector_operand_f32(state: ArchState, token: Any) -> torch.Tensor:
    if isinstance(token, str) and (token.startswith("v") or token in state.vreg):
        return state.read_vreg(token, dtype=torch.float32).clone()
    if _is_float_token(token):
        return _full_f32(state, _parse_float(token))
    return _full_f32(state, float(_parse_int(token)))


def _mask_operand(state: ArchState, token: Any) -> torch.Tensor:
    return state.read_vmreg(token) if isinstance(token, str) and token.startswith("vm") else torch.zeros(
        (state.num_sublanes, state.num_lanes),
        dtype=torch.bool,
    )


# === Address Loading Instructions ===

@instr("inlined_call_operand.hbm", BundleSlotType.SALU)
def inlined_call_operand_hbm(state: ArchState, params: SALUSlotParams):
    """Load HBM base address (in granule index, 16-byte units) for an inlined call operand.

    The parser populates args with the byte address; we store byte_addr // 16
    since subsequent sshll.u32 by 4 expects granule index.
    """
    byte_addr = params.immediate
    state.write_xreg(params.vd_reg, byte_addr)


@instr("inlined_call_operand.vmem", BundleSlotType.SALU)
def inlined_call_operand_vmem(state: ArchState, params: SALUSlotParams):
    """Load VMEM base address (byte address) for an inlined call operand.

    Parser supplies the resolved byte address; we store it directly for
    use with vld/vst and scalar_lea.vmem.
    """
    byte_addr = params.immediate
    state.write_xreg(params.vd_reg, byte_addr)


@instr("inlined_call_operand.<no memory space>", BundleSlotType.SALU)
def inlined_call_operand_smem(state: ArchState, params: SALUSlotParams):
    """Load SMEM base address (in granule index, 16-byte units) for an inlined call operand.

    Same convention as inlined_call_operand.hbm: parser passes byte address,
    we store byte_addr // 16 for sshll.u32 by 4.
    """
    byte_addr = params.immediate
    state.write_xreg(params.vd_reg, byte_addr)


@instr("scalar_parameter_address", BundleSlotType.SALU)
def scalar_parameter_address(state: ArchState, params: SALUSlotParams):
    """Load stage parameter address by index for TLP execution."""
    index = params.immediate
    runtime_params = getattr(state, "runtime_scalar_parameters", [])
    value = runtime_params[index] if 0 <= index < len(runtime_params) else 0
    state.write_xreg(params.vd_reg, value)


@instr("inlined_call", BundleSlotType.SALU)
def inlined_call(state: ArchState, params: SALUSlotParams):
    """Call control transfer is handled in Simulator."""
    return


@instr("int_to_ptr.hbm", BundleSlotType.SALU)
def int_to_ptr_hbm(state: ArchState, params: SALUSlotParams):
    """Cast an integer to an HBM (high-bandwidth memory) pointer type.

    Reads the source register (last param) and writes it to the destination
    register. The source typically holds a byte address produced by
    sshll.u32 X, 4 (granule index * 16). This instruction annotates the
    value as an HBM pointer for type checking and for consumers such as
    dma.hbm_to_vmem, dma.vmem_to_hbm, and HBM-typed loads/stores.

    In the simulator model this is a pass-through: we copy the value
    since addresses are tracked as raw integers.
    """
    src_addr_reg = params.rs1_reg
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(params.vd_reg, addr)


@instr("int_to_ptr.vmem", BundleSlotType.SALU)
def int_to_ptr_vmem(state: ArchState, params: SALUSlotParams):
    """Cast an integer to a VMEM (vector memory) pointer type.

    Reads the source register (last param) and writes it to the destination
    register. The source typically holds a byte address from sshll.u32 or
    scalar_lea.vmem. This annotates the value as a VMEM pointer for
    vld/vst and DMA operations.

    In the simulator this is a pass-through copy.
    """
    src_addr_reg = params.rs1_reg
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(params.vd_reg, addr)


# === SFlag Instructions ===

@instr("vsyncpa", BundleSlotType.SALU)
def vsyncpa(state: ArchState, params: SALUSlotParams):
    """Store a scalar value into SFlag (synchronization / scalar flag memory).

    Writes the immediate or register value to the SFlag at the given byte
    address. Used for sync flags (e.g. DMA completion), loop counters, and
    other scalar state shared across lanes. Bounds-checked against sflag_size.
    """
    if not _predicate_active(state, params):
        return
    addr = params.immediate
    value = params.rs1_reg
    addr_val = _parse_operand(state, addr)
    assert (0 <= addr_val <= state.sflag_size - torch.uint32.itemsize), f"SFLAG address out of bounds: {addr_val}"
    state.write_sflag(addr_val, _parse_operand(state, value))


@instr("vsyncadd", BundleSlotType.SALU)
def vsyncadd(state: ArchState, params: SALUSlotParams):
    """Atomic add into SFlag at the given byte address.

    Reads the current SFlag value, adds the immediate/register value (mod 256),
    and writes back. Used for reductions and lane coordination (e.g. computing
    per-lane offsets). Bounds-checked against sflag_size.
    """
    if not _predicate_active(state, params):
        return
    addr = params.immediate
    value = params.rs1_reg
    addr_val = _parse_operand(state, addr)
    if not (0 <= addr_val <= state.sflag_size - torch.uint32.itemsize):
        return
    flag_value = state.read_sflag(addr_val)
    flag_value = (flag_value + _parse_operand(state, value)) % 256
    state.write_sflag(addr_val, flag_value)


# === DMA Transfer (HBM <-> VMEM) Instructions ===

@instr("dma.hbm_to_vmem", BundleSlotType.SALU)
def dma_hbm_to_vmem(state: ArchState, params: SALUSlotParams):
    """Copy data from HBM (host memory) into VMEM (vector memory).

    Takes source and destination addresses (from int_to_ptr.hbm / int_to_ptr.vmem),
    size in granules, and a sync flag address. Copies the block and sets the
    sync flag to 1. Addresses are in granule units (16-byte); size is scaled
    internally for the memory model.
    """
    if not _predicate_active(state, params):
        return
    src_addr_reg = params.rs1_reg
    size_in_granules = params.immediate
    dest_addr_reg = params.rs2_reg
    sync_flag = params.sync_flag
    sync_flag_addr = _parse_operand(state, sync_flag)
    state.write_sflag(sync_flag_addr, 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = _parse_operand(state, size_in_granules) << 5
    state.write_vmem(dest_addr, state.read_hbm(src_addr, size))


@instr("dma.vmem_to_hbm", BundleSlotType.SALU)
def dma_vmem_to_hbm(state: ArchState, params: SALUSlotParams):
    """Copy data from VMEM (vector memory) into HBM (host memory).

    Takes source and destination addresses, size in granules, and a sync flag.
    Copies the block and sets the sync flag to 1. Used for writing results
    back to host. Address/size units match dma.hbm_to_vmem.
    """
    if not _predicate_active(state, params):
        return
    src_addr_reg = params.rs1_reg
    size_in_granules = params.immediate
    dest_addr_reg = params.rs2_reg
    sync_flag = params.sync_flag
    sync_flag_addr = _parse_operand(state, sync_flag)
    state.write_sflag(sync_flag_addr, 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = _parse_operand(state, size_in_granules) << 5
    state.write_hbm(dest_addr, state.read_vmem(src_addr, size))


@instr("dma.done.wait", BundleSlotType.SALU)
def dma_done_wait(state: ArchState, params: SALUSlotParams):
    """Wait until the DMA sync flag at the given address is set (DMA completed).

    On hardware this stalls until the corresponding dma.hbm_to_vmem or
    dma.vmem_to_hbm has finished. In this functional simulator we do not
    model stalls; the instruction is a no-op (sync flags are set immediately).
    """
    # TODO: this does not stall the execution right now
    # it simply clears the sync flag
    # need to implement this eventually
    if not _predicate_active(state, params):
        return
    return


# === Scalar Memory Load/Store Instructions ===

@instr("smov", BundleSlotType.SALU)
def smov(state: ArchState, params: SALUSlotParams):
    """Scalar move: copy immediate or register value into a scalar register.

    One operand: move that value. Two operands with predicate: pred ? b : a
    (choose b if predicate true, else a). Writes result as u32.
    """
    value_a = params.rs1_reg
    value_b = params.rs2_reg
    if not params.predication:
        value = _parse_operand(state, value_a)
    else:
        # TPU predicated scalar move form behaves as:
        #   smov (pred, a), b  => pred ? b : a
        active = _predicate_active(state, params)
        on_pred = value_a
        fallback = value_b
        chosen = fallback if active else on_pred
        value = _parse_operand(state, chosen)
    state.write_xreg(params.vd_reg, _as_u32(value))


# === SALU Instructions ===

@instr("sshll.u32", BundleSlotType.SALU)
def sshll_u32(state: ArchState, params: SALUSlotParams):
    """Scalar shift left logical: dest = src << imm (u32).

    Commonly used with imm=4 to convert granule index to byte address
    (multiply by 16) before int_to_ptr.hbm or int_to_ptr.vmem.
    """
    if not _predicate_active(state, params):
        return
    src_reg = params.rs1_reg
    imm = params.immediate
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, src_reg) << imm))


@instr("sshra.s32", BundleSlotType.SALU)
def sshra_s32(state: ArchState, params: SALUSlotParams):
    """Scalar shift right arithmetic: dest = src >> imm (s32, sign-extended).

    Used for pointer arithmetic or dividing signed values by powers of two.
    """
    if not _predicate_active(state, params):
        return
    src_reg = params.rs1_reg
    imm = params.immediate
    value = _as_i32(_parse_operand(state, src_reg))
    state.write_xreg(params.vd_reg, _as_u32(value >> imm))


@instr("sshrl.u32", BundleSlotType.SALU)
def sshrl_u32(state: ArchState, params: SALUSlotParams):
    """Scalar shift right logical: dest = src >> imm (u32)."""
    if not _predicate_active(state, params):
        return
    src_reg = params.rs1_reg
    imm = params.immediate
    value = _as_u32(_parse_operand(state, src_reg))
    state.write_xreg(params.vd_reg, _as_u32(value >> (imm & 31)))


@instr("sadd.s32", BundleSlotType.SALU)
def sadd_s32(state: ArchState, params: SALUSlotParams):
    """Scalar add: dest = a + b (s32, truncated to u32)."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) + _parse_operand(state, b)))


@instr("ssub.s32", BundleSlotType.SALU)
def ssub_s32(state: ArchState, params: SALUSlotParams):
    """Scalar subtract: dest = a - b (s32, truncated to u32)."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) - _parse_operand(state, b)))


@instr("sor.u32", BundleSlotType.SALU)
def sor_u32(state: ArchState, params: SALUSlotParams):
    """Scalar bitwise or: dest = a | b (u32)."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) | _parse_operand(state, b)))


@instr("sand.u32", BundleSlotType.SALU)
def sand_u32(state: ArchState, params: SALUSlotParams):
    """Scalar bitwise and: dest = a & b (u32)."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) & _parse_operand(state, b)))


@instr("smul.u32", BundleSlotType.SALU)
def smul_u32(state: ArchState, params: SALUSlotParams):
    """Scalar multiply (u32 wraparound): dest = a * b."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) * _parse_operand(state, b)))


@instr("sxor.u32", BundleSlotType.SALU)
def sxor_u32(state: ArchState, params: SALUSlotParams):
    """Scalar bitwise xor: dest = a ^ b (u32)."""
    if not _predicate_active(state, params):
        return
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, a) ^ _parse_operand(state, b)))


@instr("scalar_select", BundleSlotType.SALU)
def scalar_select(state: ArchState, params: SALUSlotParams):
    """Select scalar by predicate: dest = pred ? on_true : on_false.

    Reads the predicate register and writes the chosen scalar value.
    """
    pred_reg = params.pred
    on_true = params.rs1_reg
    on_false = params.rs2_reg
    chosen = on_true if state.read_preg(pred_reg) else on_false
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, chosen)))


@instr("sphi", BundleSlotType.SALU)
def sphi(state: ArchState, params: SALUSlotParams):
    """Scalar phi (SSA merge): pick value based on control flow.

    In SSA form this would merge values from different predecessors. After
    register coalescing in the compiler, it reduces to a move.
    """
    src = params.rs1_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, src)))


@instr("scmp.eq.s32.totalorder", BundleSlotType.SALU)
def scmp_eq_s32_totalorder(state: ArchState, params: SALUSlotParams):
    """Scalar compare equal: dest_pred = (a == b) as s32."""
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_preg(params.vd_reg, _as_i32(_parse_operand(state, a)) == _as_i32(_parse_operand(state, b)))


@instr("scmp.ne.s32.totalorder", BundleSlotType.SALU)
def scmp_ne_s32_totalorder(state: ArchState, params: SALUSlotParams):
    """Scalar compare not-equal: dest_pred = (a != b) as s32."""
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_preg(params.vd_reg, _as_i32(_parse_operand(state, a)) != _as_i32(_parse_operand(state, b)))


@instr("scmp.ge.s32.totalorder", BundleSlotType.SALU)
def scmp_ge_s32_totalorder(state: ArchState, params: SALUSlotParams):
    """Scalar compare greater-or-equal: dest_pred = (a >= b) as s32."""
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_preg(params.vd_reg, _as_i32(_parse_operand(state, a)) >= _as_i32(_parse_operand(state, b)))


@instr("scmp.lt.s32.totalorder", BundleSlotType.SALU)
def scmp_lt_s32_totalorder(state: ArchState, params: SALUSlotParams):
    """Scalar compare less-than: dest_pred = (a < b) as s32."""
    a = params.rs1_reg
    b = params.rs2_reg
    state.write_preg(params.vd_reg, _as_i32(_parse_operand(state, a)) < _as_i32(_parse_operand(state, b)))


@instr("por", BundleSlotType.SALU)
def por(state: ArchState, params: SALUSlotParams):
    """Predicate or: dest = p0 or p1."""
    p0 = params.ps1_reg
    p1 = params.ps2_reg
    state.write_preg(params.vd_reg, state.read_preg(p0) or state.read_preg(p1))


@instr("pnand", BundleSlotType.SALU)
def pnand(state: ArchState, params: SALUSlotParams):
    """Predicate nand: dest = p0 and (not p1)."""
    p0 = params.ps1_reg
    p1 = params.ps2_reg
    state.write_preg(params.vd_reg, state.read_preg(p0) and (not state.read_preg(p1)))


@instr("pneg", BundleSlotType.SALU)
def pneg(state: ArchState, params: SALUSlotParams):
    """Predicate negate: dest = not p0."""
    p0 = params.ps1_reg
    state.write_preg(params.vd_reg, not state.read_preg(p0))


@instr("scalar_lea.vmem", BundleSlotType.SALU)
def scalar_lea_vmem(state: ArchState, params: SALUSlotParams):
    """Load-effective-address for VMEM: dest = base + (offset << 9).

    Computes VMEM byte address from base and scaled offset. Used for
    indexing into vector memory with tile or row strides.
    """
    if not _predicate_active(state, params):
        return
    base = params.rs1_reg
    offset = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.hbm", BundleSlotType.SALU)
def scalar_lea_hbm(state: ArchState, params: SALUSlotParams):
    """Load-effective-address for HBM: dest = base + (offset << 9).

    Computes HBM byte address from base and scaled offset.
    """
    if not _predicate_active(state, params):
        return
    base = params.rs1_reg
    offset = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.sflag", BundleSlotType.SALU)
def scalar_lea_sflag(state: ArchState, params: SALUSlotParams):
    """Load-effective-address for SFlag: dest = base + (offset << 2).

    Offset scaled by 4 (uint32 size) for indexing SFlag entries.
    """
    if not _predicate_active(state, params):
        return
    base = params.rs1_reg
    offset = params.rs2_reg
    state.write_xreg(params.vd_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 2)))


@instr("sst", BundleSlotType.SALU)
def sst(state: ArchState, params: SALUSlotParams):
    """Scalar store: write 4-byte value from src_reg to SMEM at smem_addr.

    SMEM address is in bytes. Little-endian.
    """
    smem_addr = params.address
    src_reg = params.rs1_reg
    address = _parse_operand(state, smem_addr)
    value = _as_u32(_parse_operand(state, src_reg))
    raw = torch.tensor(list(value.to_bytes(4, byteorder="little", signed=False)), dtype=torch.uint8)
    state.write_smem(address, raw)


@instr("sld", BundleSlotType.SALU)
def sld(state: ArchState, params: SALUSlotParams):
    """Scalar load: read 4 bytes from SMEM at smem_addr into dest_reg.

    Little-endian. Used for loading constants or scalar values from SMEM.
    """
    smem_addr = params.address
    address = _parse_operand(state, smem_addr)
    raw = state.read_smem(address, 4, dtype=torch.uint8).tolist()
    state.write_xreg(params.vd_reg, int.from_bytes(bytes(raw), byteorder="little", signed=False))


@instr("sbr.rel", BundleSlotType.SALU)
def sbr_rel(state: ArchState, params: SALUSlotParams):
    """Relative scalar branch: set PC to target (bundle index)."""
    if not _predicate_active(state, params):
        return
    state.next_pc = params.target


@instr("shalt.err", BundleSlotType.SALU)
def shalt_err(state: ArchState, params: SALUSlotParams):
    """Halt on error: predicate-guarded trap for bounds checks or assertions.

    On hardware would halt the core. In this functional simulator we treat
    it as non-fatal (no-op) for continued execution.
    """
    if not _predicate_active(state, params):
        return
    # Bounds checks are modeled as non-fatal in this functional simulator.
    return


# === Tensor Memory Load/Store Instructions ===

@instr("vstv", BundleSlotType.SALU)
def vstv(state: ArchState, params: SALUSlotParams):
    """Load and broadcast a scalar from SMEM into a vector register.

    Reads 4 bytes (float32) at the address in src_reg and replicates it
    across all sublanes and lanes. Used for broadcasting constants into
    vector ops (e.g. before vpack for BF16 conversion).
    """
    src_reg = params.rs1_reg
    address = state.read_xreg(src_reg)
    scalar_data = state.read_smem(address, 4, dtype=torch.float32)
    data = torch.tensor([scalar_data], dtype=torch.float32).repeat(state.num_sublanes, state.num_lanes)
    state.write_vreg(params.vd_reg, data)


@instr("vld", BundleSlotType.LOAD)
def vld(state: ArchState, params: LOADParams):
    """Vector load from VMEM into a vector register.

    Address can be register, register+offset, or immediate. Optional
    sublane mask (8-bit) zeroes out masked rows. Loads vreg_size bytes
    and reshapes to num_sublanes x lanes.
    """
    address = _parse_operand(state, params.address)
    ss_stride = params.sublane_stride if isinstance(params.sublane_stride, int) else None

    data = state.read_vmem(address, state.vreg_size).reshape(state.num_sublanes, -1)

    # ss:$0 means all sublanes read from the same base row.
    if ss_stride == 0:
        data = data[0:1, :].repeat(state.num_sublanes, 1)

    mask_val = int(params.sublane_mask)
    if mask_val != 255:
        # 8-bit mask: bit i = 1 keep row i, bit i = 0 clear row i to zero
        row_mask = torch.tensor(
            [(mask_val >> i) & 1 for i in range(state.num_sublanes)],
            dtype=torch.bool,
            device=data.device,
        ).unsqueeze(1)
        data = data * row_mask

        if state.verbose:
            print(f"\033[90m  Load with mask '{mask_val}' -> {row_mask.flatten().int().tolist()}\033[0m")

    state.write_vreg(params.vd_reg, data)


@instr("vst", BundleSlotType.STORE)
def vst(state: ArchState, params: STOREParams):
    """Vector store from a vector register to VMEM.

    Address (register or immediate), optional sublane mask (8-bit,
    bit i = 1 stores row i), and source register. Mask 0 stores nothing.
    """
    address = _parse_operand(state, params.address)
    vsrc_reg = params.vs1_reg
    data = state.read_vreg(vsrc_reg, dtype=torch.uint8)
    mask_val = int(params.sublane_mask)
    if mask_val == 255:
        state.write_vmem(address, data.flatten())
        return
    if mask_val == 0:
        if state.verbose:
            print("\033[90m  Store with mask '0' -> [] (no rows stored)\033[0m")
        return

    row_mask = torch.tensor(
        [(mask_val >> i) & 1 for i in range(state.num_sublanes)],
        dtype=torch.bool,
        device=data.device,
    ).unsqueeze(1)
    existing = state.read_vmem(address, state.vreg_size).reshape(state.num_sublanes, -1)
    merged = torch.where(row_mask, data, existing)
    state.write_vmem(address, merged.flatten())


@instr("vst.msk", BundleSlotType.STORE)
def vst_msk(state: ArchState, params: STOREParams):
    """Vector store to VMEM with sublane mask (dest_reg variant).

    Same as vst but with a dedicated mask form; bit i = 1 stores row i.
    """
    address = _parse_operand(state, params.address)
    vsrc_reg = params.vs1_reg
    data = state.read_vreg(vsrc_reg, dtype=torch.uint8)
    mask_val = int(params.sublane_mask)
    if mask_val == 255:
        state.write_vmem(address, data.flatten())
        return
    if mask_val == 0:
        if state.verbose:
            print("\033[90m  Store with mask '0' -> [] (no rows stored)\033[0m")
        return

    row_mask = torch.tensor(
        [(mask_val >> i) & 1 for i in range(state.num_sublanes)],
        dtype=torch.bool,
        device=data.device,
    ).unsqueeze(1)
    existing = state.read_vmem(address, state.vreg_size).reshape(state.num_sublanes, -1)
    merged = torch.where(row_mask, data, existing)
    state.write_vmem(address, merged.flatten())


# === VPU Instructions ===

@instr("vadd.f32", BundleSlotType.VALU)
def vadd_f32(state: ArchState, params: VALUSlotParams):
    """Vector add (float32): dest = vsrc1 + vsrc2 elementwise.

    Operands can be vector registers or immediates.
    """
    vsrc1_reg = params.vs1_reg
    vsrc2_reg = params.vs2_reg
    if isinstance(vsrc1_reg, str) and vsrc1_reg.startswith("v"):
        vsrc1_data = state.read_vreg(vsrc1_reg, dtype=torch.float32)
    else:
        vsrc1_data = float(vsrc1_reg)
    if isinstance(vsrc2_reg, str) and vsrc2_reg.startswith("v"):
        vsrc2_data = state.read_vreg(vsrc2_reg, dtype=torch.float32)
    else:
        vsrc2_data = float(vsrc2_reg)

    result = vsrc1_data + vsrc2_data
    state.write_vreg(params.vd_reg, result)


@instr("vmov", BundleSlotType.VALU)
def vmov(state: ArchState, params: VALUSlotParams):
    """Broadcast scalar immediate/register value into a vector register."""
    src = params.vs1_reg
    if isinstance(src, str) and (src.startswith("v") or src in state.vreg):
        state.write_vreg(params.vd_reg, state.read_vreg(src, dtype=torch.float32))
        return
    if _is_float_token(src):
        state.write_vreg(params.vd_reg, _full_f32(state, _parse_float(src)))
        return
    state.write_vreg(params.vd_reg, _full_u32(state, _parse_int(src)))


@instr("vadd.s32", BundleSlotType.VALU)
def vadd_s32(state: ArchState, params: VALUSlotParams):
    """Vector add in 32-bit integer lanes with wraparound."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(params.vd_reg, ((lhs + rhs) & U32_MASK).to(torch.uint32))


@instr("vsub.s32", BundleSlotType.VALU)
def vsub_s32(state: ArchState, params: VALUSlotParams):
    """Vector subtract in 32-bit integer lanes with wraparound."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(params.vd_reg, ((lhs - rhs) & U32_MASK).to(torch.uint32))


@instr("vsub.f32", BundleSlotType.VALU)
def vsub_f32(state: ArchState, params: VALUSlotParams):
    """Vector float subtraction."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vreg(params.vd_reg, _vector_operand_f32(state, a) - _vector_operand_f32(state, b))


@instr("vmul.f32", BundleSlotType.VALU)
def vmul_f32(state: ArchState, params: VALUSlotParams):
    """Vector float multiplication."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vreg(params.vd_reg, _vector_operand_f32(state, a) * _vector_operand_f32(state, b))


@instr("vmul.u32", BundleSlotType.VALU)
def vmul_u32(state: ArchState, params: VALUSlotParams):
    """Vector 32-bit unsigned multiply with wraparound."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(params.vd_reg, ((lhs * rhs) & U32_MASK).to(torch.uint32))


@instr("vand.u32", BundleSlotType.VALU)
def vand_u32(state: ArchState, params: VALUSlotParams):
    """Vector bitwise AND on 32-bit lanes."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vreg(params.vd_reg, _vector_operand_u32(state, a) & _vector_operand_u32(state, b))


@instr("vor.u32", BundleSlotType.VALU)
def vor_u32(state: ArchState, params: VALUSlotParams):
    """Vector bitwise OR on 32-bit lanes."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vreg(params.vd_reg, _vector_operand_u32(state, a) | _vector_operand_u32(state, b))


@instr("vxor.u32", BundleSlotType.VALU)
def vxor_u32(state: ArchState, params: VALUSlotParams):
    """Vector bitwise XOR on 32-bit lanes."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vreg(params.vd_reg, _vector_operand_u32(state, a) ^ _vector_operand_u32(state, b))


@instr("vshll.u32", BundleSlotType.VALU)
def vshll_u32(state: ArchState, params: VALUSlotParams):
    """Vector logical left shift on 32-bit lanes."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    sh = (_vector_operand_u32(state, b) & 31).to(torch.int64)
    state.write_vreg(params.vd_reg, ((lhs << sh) & U32_MASK).to(torch.uint32))


@instr("vshrl.u32", BundleSlotType.VALU)
def vshrl_u32(state: ArchState, params: VALUSlotParams):
    """Vector logical right shift on 32-bit lanes."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    sh = (_vector_operand_u32(state, b) & 31).to(torch.int64)
    state.write_vreg(params.vd_reg, ((lhs >> sh) & U32_MASK).to(torch.uint32))


@instr("vclz", BundleSlotType.VALU)
def vclz(state: ArchState, params: VALUSlotParams):
    """Vector count-leading-zeros for 32-bit lanes."""
    src = params.vs1_reg
    values = _vector_operand_u32(state, src).flatten().tolist()
    clz = [
        (32 - int(v).bit_length()) if int(v) != 0 else 32
        for v in values
    ]
    result = torch.tensor(clz, dtype=torch.uint32).reshape(state.num_sublanes, state.num_lanes)
    state.write_vreg(params.vd_reg, result)


@instr("vcvt.s32.f32", BundleSlotType.VALU)
def vcvt_s32_f32(state: ArchState, params: VALUSlotParams):
    """Convert vector int32 lanes to float32 lanes."""
    src = params.vs1_reg
    state.write_vreg(params.vd_reg, _vector_operand_i32(state, src).to(torch.float32))


@instr("vcvt.f32.f8e4m3b11", BundleSlotType.VALU)
def vcvt_f32_f8e4m3b11(state: ArchState, params: VALUSlotParams):
    """Quantize float32 lanes to F8 and return packed 8-bit codes in u32 lanes."""
    src = params.vs1_reg
    f8 = _vector_operand_f32(state, src).to(torch.float8_e4m3fn)
    codes = f8.view(torch.uint8).to(torch.uint32).contiguous()
    state.write_vreg(params.vd_reg, codes)


@instr("vcmp.lt.s32.totalorder", BundleSlotType.VALU)
def vcmp_lt_s32_totalorder(state: ArchState, params: VALUSlotParams):
    """Vector signed int compare (<), producing VM mask."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vmreg(params.vd_reg, _vector_operand_i32(state, a) < _vector_operand_i32(state, b))


@instr("vcmp.gt.s32.totalorder", BundleSlotType.VALU)
def vcmp_gt_s32_totalorder(state: ArchState, params: VALUSlotParams):
    """Vector signed int compare (>), producing VM mask."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vmreg(params.vd_reg, _vector_operand_i32(state, a) > _vector_operand_i32(state, b))


@instr("vcmp.eq.s32.totalorder", BundleSlotType.VALU)
def vcmp_eq_s32_totalorder(state: ArchState, params: VALUSlotParams):
    """Vector signed int compare (==), producing VM mask."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vmreg(params.vd_reg, _vector_operand_i32(state, a) == _vector_operand_i32(state, b))


@instr("vcmp.le.f32.partialorder", BundleSlotType.VALU)
def vcmp_le_f32_partialorder(state: ArchState, params: VALUSlotParams):
    """Vector float compare (<=) with partial-order NaN handling."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs <= rhs)
    state.write_vmreg(params.vd_reg, mask)


@instr("vcmp.eq.f32.partialorder", BundleSlotType.VALU)
def vcmp_eq_f32_partialorder(state: ArchState, params: VALUSlotParams):
    """Vector float compare (==) with partial-order NaN handling."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs == rhs)
    state.write_vmreg(params.vd_reg, mask)


@instr("vcmp.gt.f32.partialorder", BundleSlotType.VALU)
def vcmp_gt_f32_partialorder(state: ArchState, params: VALUSlotParams):
    """Vector float compare (>) with partial-order NaN handling."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs > rhs)
    state.write_vmreg(params.vd_reg, mask)


@instr("vcmp.ne.f32.partialorder", BundleSlotType.VALU)
def vcmp_ne_f32_partialorder(state: ArchState, params: VALUSlotParams):
    """Vector float compare (!=) with partial-order NaN handling."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs != rhs)
    state.write_vmreg(params.vd_reg, mask)


@instr("vc.u32", BundleSlotType.VALU)
def vc_u32(state: ArchState, params: VALUSlotParams):
    """Vector carry-out mask for 32-bit unsigned addition."""
    a = params.vs1_reg
    b = params.vs2_reg
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vmreg(params.vd_reg, (lhs + rhs) > U32_MASK)


@instr("vmor", BundleSlotType.VALU)
def vmor(state: ArchState, params: VALUSlotParams):
    """Mask OR."""
    a = params.vs1_reg
    b = params.vs2_reg
    state.write_vmreg(params.vd_reg, _mask_operand(state, a) | _mask_operand(state, b))


@instr("vsel", BundleSlotType.VALU)
def vsel(state: ArchState, params: VALUSlotParams):
    """Vector select by VM mask: mask ? on_true : on_false."""
    vm_reg = params.vm_reg
    on_true = params.vs1_reg
    on_false = params.vs2_reg
    mask = _mask_operand(state, vm_reg)
    true_bits = _vector_operand_u32(state, on_true)
    false_bits = _vector_operand_u32(state, on_false)
    selected = torch.where(mask, true_bits.to(torch.int64), false_bits.to(torch.int64)).to(torch.uint32)
    state.write_vreg(params.vd_reg, selected)


@instr("vlaneseq", BundleSlotType.VALU)
def vlaneseq(state: ArchState, params: VALUSlotParams):
    """Lane index sequence as a linear register-space id.

    Hardware semantics for emitted iota kernels require values to advance
    across both lane and sublane dimensions so a single vreg carries
    [0..num_sublanes*num_lanes-1] in row-major order.
    """
    base = (
        torch.arange(state.num_sublanes, dtype=torch.int64)
        .unsqueeze(1)
        .mul(state.num_lanes)
    )
    lane = torch.arange(state.num_lanes, dtype=torch.int64).unsqueeze(0)
    seq = (base + lane).to(torch.uint32).contiguous()
    state.write_vreg(params.vd_reg, seq)


@instr("vset.pattern.permute.xlu0", BundleSlotType.XLU)
def vset_pattern_permute_xlu0(state: ArchState, params: XLUSlotParams):
    """Record permute pattern token (functional model)."""
    # Pattern metadata is not fully modeled; keep the latest token around so
    # vperm/vpop.permute can coordinate temporary ids.
    if params.vd_reg:
        state.last_permute_token = params.vd_reg


@instr("vperm.xlu0", BundleSlotType.XLU)
def vperm_xlu0(state: ArchState, params: XLUSlotParams):
    """Queue a source vector for a subsequent vpop.permute.xlu0."""
    source = params.vs1_reg if isinstance(params.vs1_reg, str) and params.vs1_reg.startswith("v") else None
    if source is None:
        return
    token = params.vd_reg if params.vd_reg else f"perm_{len(state.permute_buffer)}"
    state.permute_buffer[token] = state.read_vreg(source, dtype=torch.float32).clone()
    state.last_permute_token = token


def _permute_token(params: XLUSlotParams, state: ArchState) -> str | None:
    token = params.immediate if params.has_imm1 else state.last_permute_token
    if token is None or (isinstance(token, int) and token == 0):
        token = state.last_permute_token
    if token is None:
        return None
    return str(token)


@instr("vpop.permute.xlu0", BundleSlotType.XLU)
def vpop_permute_xlu0(state: ArchState, params: XLUSlotParams):
    """Pop permuted data from the functional permute buffer.

    Current model covers the emitted TPU patterns used by these kernels:
    source vectors of logical shape [8] are expanded into per-row broadcasts.
    """
    token = _permute_token(params, state)
    if token is None or token not in state.permute_buffer:
        state.write_vreg(params.vd_reg, torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32))
        return
    src = state.permute_buffer[token]
    vec8 = src[0, :state.num_sublanes]
    out = vec8.unsqueeze(1).repeat(1, state.num_lanes).contiguous()
    state.write_vreg(params.vd_reg, out)


@instr("vpop.permute.xlu1", BundleSlotType.XLU)
def vpop_permute_xlu1(state: ArchState, params: XLUSlotParams):
    """Pop a value from the permute token buffer (xlu1 variant)."""
    token = _permute_token(params, state)
    if token is None or token not in state.permute_buffer:
        state.write_vreg(params.vd_reg, torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32))
        return
    state.write_vreg(params.vd_reg, state.permute_buffer[token].clone())


@instr("vpop.permute.xlu2", BundleSlotType.XLU)
def vpop_permute_xlu2(state: ArchState, params: XLUSlotParams):
    """Pop a value from the permute token buffer (xlu2 variant)."""
    token = _permute_token(params, state)
    if token is None or token not in state.permute_buffer:
        state.write_vreg(params.vd_reg, torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32))
        return
    state.write_vreg(params.vd_reg, state.permute_buffer[token].clone())


@instr("vrcp.f32", BundleSlotType.VALU)
def vrcp_f32(state: ArchState, params: VALUSlotParams):
    """Vector reciprocal approximation (modeled as exact reciprocal)."""
    src = params.vs1_reg
    state.write_vreg(params.vd_reg, 1.0 / _vector_operand_f32(state, src))


@instr("vpow2.f32", BundleSlotType.VALU)
def vpow2_f32(state: ArchState, params: VALUSlotParams):
    """Vector base-2 exponent."""
    src = params.vs1_reg
    state.write_vreg(params.vd_reg, torch.pow(2.0, _vector_operand_f32(state, src)))


@instr("vpop.eup", BundleSlotType.EUP)
def vpop_eup(state: ArchState, params: EUPParams):
    """Pop EUP pipeline value (modeled as identity)."""
    src = params.vs1_reg
    state.write_vreg(params.vd_reg, _vector_operand_f32(state, src))


@instr("vweird.f32", BundleSlotType.VALU)
def vweird_f32(state: ArchState, params: VALUSlotParams):
    """Special-value detector for transcendental pipelines."""
    src = params.vs1_reg
    data = _vector_operand_f32(state, src)
    state.write_vmreg(params.vd_reg, ~torch.isfinite(data))


@instr("vtanh.f32", BundleSlotType.VALU)
def vtanh_f32(state: ArchState, params: VALUSlotParams):
    """Vector tanh."""
    src = params.vs1_reg
    state.write_vreg(params.vd_reg, torch.tanh(_vector_operand_f32(state, src)))


@instr("vrot.slane", BundleSlotType.VALU)
def vrot_slane(state: ArchState, params: VALUSlotParams):
    """Rotate vector rows (sublanes) by shift_amount positions.

    Rolls the sublane dimension; used for data movement in reductions
    and lane communication.
    """
    vsrc_reg = params.vs1_reg
    shift_amount = params.vs1_imm
    assert shift_amount >= 0 and shift_amount < state.num_sublanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    vdest_data = vsrc_data.roll(shift_amount, dims=0)
    state.write_vreg(params.vd_reg, vdest_data)


def _vrot_lane_b32(state: ArchState, params: XLUSlotParams):
    src = params.vs1_reg
    shift_token = params.rs1_reg if params.has_rs1 and params.rs1_reg else params.immediate
    shift = _parse_operand(state, shift_token) % state.num_lanes
    rotated = _vector_operand_f32(state, src).roll(shifts=shift, dims=1).contiguous()
    token = params.vd_reg if params.vd_reg else f"perm_{len(state.permute_buffer)}"
    state.permute_buffer[token] = rotated
    state.last_permute_token = token


@instr("vrot.lane.b32.xlu0", BundleSlotType.XLU)
def vrot_lane_b32_xlu0(state: ArchState, params: XLUSlotParams):
    """Lane rotation feeding XLU0 permute path."""
    _vrot_lane_b32(state, params)


@instr("vrot.lane.b32.xlu1", BundleSlotType.XLU)
def vrot_lane_b32_xlu1(state: ArchState, params: XLUSlotParams):
    """Lane rotation feeding XLU1 permute path."""
    _vrot_lane_b32(state, params)


@instr("vrot.lane.b32.xlu2", BundleSlotType.XLU)
def vrot_lane_b32_xlu2(state: ArchState, params: XLUSlotParams):
    """Lane rotation feeding XLU2 permute path."""
    _vrot_lane_b32(state, params)


@instr("vcmask", BundleSlotType.VALU)
def vcmask(state: ArchState, params: VALUSlotParams):
    """Configure vector mask (submatrix selection for loads/stores).

    Vector mask registers are not explicitly modeled in this simulator;
    the instruction is a no-op.
    """
    return


# === Tensor Packing/Unpacking Instructions ===

@instr("vpack.c.bf16", BundleSlotType.VALU)
def vpack_c_bf16(state: ArchState, params: VALUSlotParams):
    """Pack two float32 registers (high, low) into one BF16 register.

    Concatenates high and low as BF16; typically used after vadd/vstv
    to prepare data for MXU matmul.
    """
    reg1_or_imm = params.vs1_reg
    vsrc2_reg = params.vs2_reg
    if isinstance(reg1_or_imm, str) and reg1_or_imm.startswith("v"):
        high = state.read_vreg(reg1_or_imm, dtype=torch.float32)
    else:
        high = torch.full(
            (state.num_sublanes, state.num_lanes),
            fill_value=float(reg1_or_imm),
            dtype=torch.float32,
        )
    low = state.read_vreg(vsrc2_reg, dtype=torch.float32)
    packed = pack_bf16_register(
        low=low.to(torch.bfloat16),
        high=high.to(torch.bfloat16),
        num_sublanes=state.num_sublanes,
        num_lanes=state.num_lanes,
    )
    state.write_vreg(params.vd_reg, packed)


@instr("vpack.c.b16", BundleSlotType.VALU)
def vpack_c_b16(state: ArchState, params: VALUSlotParams):
    """Pack two 32-bit sources into one 16-bit-packed register image."""
    high_token = params.vs1_reg
    low = (_vector_operand_u32(state, params.vs2_reg) & 0xFFFF).to(torch.uint16)
    if not (isinstance(high_token, str) and high_token.startswith("v")) and _parse_int(high_token) == 0:
        # Emitted f8 path: fold 8x128 source rows into 4 rows of 256 lanes.
        packed = torch.zeros((state.num_sublanes, state.num_lanes * 2), dtype=torch.uint16)
        for i in range(state.num_sublanes // 2):
            packed[i, :state.num_lanes] = low[2 * i, :]
            packed[i, state.num_lanes:] = low[2 * i + 1, :]
        state.write_vreg(params.vd_reg, packed.contiguous())
        return
    high = (_vector_operand_u32(state, high_token) & 0xFFFF).to(torch.uint16)
    packed = torch.cat([low, high], dim=1).contiguous()
    state.write_vreg(params.vd_reg, packed)


@instr("vpack.c.b8", BundleSlotType.VALU)
def vpack_c_b8(state: ArchState, params: VALUSlotParams):
    """Pack two 16-bit sources into one 8-bit-packed register image."""
    high_token = params.vs1_reg
    low = (_vector_operand_u16(state, params.vs2_reg) & 0xFF).to(torch.uint8)
    if not (isinstance(high_token, str) and high_token.startswith("v")) and _parse_int(high_token) == 0:
        # Emitted f8 path: fold 4x256 rows into 2 rows of 512 lanes.
        packed = torch.zeros((state.num_sublanes, state.num_lanes * 4), dtype=torch.uint8)
        packed[0, :state.num_lanes*2] = low[0, :]
        packed[0, state.num_lanes*2:] = low[1, :]
        packed[1, :state.num_lanes*2] = low[2, :]
        packed[1, state.num_lanes*2:] = low[3, :]
        state.write_vreg(params.vd_reg, packed.contiguous())
        return
    high = (_vector_operand_u16(state, high_token) & 0xFF).to(torch.uint8)
    packed = torch.cat([low, high], dim=1).contiguous()
    state.write_vreg(params.vd_reg, packed)


@instr("vunpack.c.0.f8e4m3b11", BundleSlotType.VALU)
def vunpack_c_0_f8e4m3b11(state: ArchState, params: VALUSlotParams):
    """Unpack lane-0 bytes of packed F8 data into float32 lanes."""
    src = params.vs1_reg
    packed_u8 = state.read_vreg(src, dtype=torch.uint8)
    # Packed F8 inputs are laid out as 1024 lane-0 bytes in the leading
    # segment of the 4096-byte register image (commonly loaded with sm:$0x3).
    lane0 = packed_u8.flatten()[: state.num_sublanes * state.num_lanes].reshape(
        state.num_sublanes, state.num_lanes
    ).contiguous()
    unpacked = lane0.view(torch.float8_e4m3fn).to(torch.float32).contiguous()
    state.write_vreg(params.vd_reg, unpacked)


@instr("vunpack.c.l.bf16", BundleSlotType.VALU)
def vunpack_c_l_bf16(state: ArchState, params: VALUSlotParams):
    """Unpack low half of packed BF16 register to float32.

    Extracts the lower 16 bits of each 32-bit lane; used after MXU
    output to convert BF16 results back to FP32.
    """
    vsrc_reg = params.vs1_reg
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    low, _ = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(params.vd_reg, low.to(torch.float32))


@instr("vunpack.c.h.bf16", BundleSlotType.VALU)
def vunpack_c_h_bf16(state: ArchState, params: VALUSlotParams):
    """Unpack high half of packed BF16 register to float32.

    Extracts the upper 16 bits of each 32-bit lane.
    """
    vsrc_reg = params.vs1_reg
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    _, high = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(params.vd_reg, high.to(torch.float32))


@instr("vunpack.i.l.bf16", BundleSlotType.VALU)
def vunpack_i_l_bf16(state: ArchState, params: VALUSlotParams):
    """Unpack low half (immediate path): reinterpret FP32 lanes as BF16 low.

    For values from scalar path (e.g. vstv) represented as FP32; converts
    to BF16 and back to FP32 for downstream use.
    """
    vsrc_reg = params.vs1_reg
    # Immediate BF16 values originate from scalar paths (e.g. vstv), where values
    # are represented in FP32 lanes and then interpreted as BF16 for conversion.
    unpacked_data = state.read_vreg(vsrc_reg, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    state.write_vreg(params.vd_reg, unpacked_data.contiguous())


# === XLU Instructions ===


@instr("vxpose.xlu0.b32.start.end", BundleSlotType.XLU)
def vxpose_xlu0_b32_start_end(state: ArchState, params: XLUSlotParams):
    """
    Transpose the source register content and load it into the XLU unit.

    The XLU buffer is a 128 x 16 array. Tensor registers are first transposed and then
    shifted into the buffer from bottom to top (decreasing row index). The result
    is read out from the top-most 8 rows from left to right out of the unit, to form the
    leftmost 8 x 8 submatrix of the destination register, with rest of the elements in
    the destination register set to zero. The rest of the content in the XLU is shifted
    upwards. This effectively slides a 8 x 8 window downward over the buffer.

    XLU buffer (128 x 16):
            +--------+
    row 0   |XXXXXXXX| -- >
            |XXXXXXXX| -- >  (8 x 8) submatrix
            |XXXXXXXX| -- >
    row 7   |XXXXXXXX| -- >
            +--------+
    row 8   |        |
            |        |
            |        |  ^
            |        |  |
            |        |  |
            |  XLU   |  |
            | Buffer |  |
            |        |  |
            |        |  |
            |        |  |
            |        |
    row 127 |        |
            +--------+
            |        |
            |        |
             ^ ^ ^ ^
             | | | |
            (128 x 8) transposed matrix register
    """
    vsrc_reg = params.vs1_reg
    num_lanes = params.immediate
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    transposed = vsrc_data.transpose(0, 1).contiguous()
    state.xlu_buffer[0:num_lanes, :] = transposed[0:num_lanes, :]
    state.xlu_pop_width = state.num_sublanes


@instr("vxpose.xlu0.b32.start", BundleSlotType.XLU)
def vxpose_xlu0_b32_start(state: ArchState, params: XLUSlotParams):
    """Transpose and load into XLU buffer (start of sequence).

    Transposes the source tensor (num_lanes x sublane) and stores it in
    the top rows of the XLU buffer. Used with vxpose.xlu0.b32.end for
    multi-tile transpose.
    """
    vsrc_reg = params.vs1_reg
    num_lanes = params.immediate
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    transposed = vsrc_data.transpose(0, 1).contiguous()
    state.xlu_buffer[0:num_lanes, :] = transposed[0:num_lanes, :]
    state.xlu_pop_width = state.num_sublanes


@instr("vxpose.xlu0.b32.end", BundleSlotType.XLU)
def vxpose_xlu0_b32_end(state: ArchState, params: XLUSlotParams):
    """Transpose and load into XLU buffer (end of sequence).

    Rolls the buffer up, then writes transposed source into the next
    rows. Complements vxpose.xlu0.b32.start for sliding-window transpose.
    """
    vsrc_reg = params.vs1_reg
    num_lanes = params.immediate
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    transposed = vsrc_data.transpose(0, 1).contiguous()
    # For a start/end pair, append into the second half of the XLU staging
    # buffer so subsequent vpop.trf can consume both halves sequentially.
    state.xlu_buffer[state.num_lanes:state.num_lanes + num_lanes, :] = transposed[0:num_lanes, :]
    state.xlu_pop_width = state.num_sublanes * 2


@instr("vxpose.xlu0.b32.cont", BundleSlotType.XLU)
def vxpose_xlu0_b32_cont(state: ArchState, params: XLUSlotParams):
    """Continuation marker for XLU transpose sequence (no-op in this model)."""
    return


@instr("vpop.trf.xlu0", BundleSlotType.XLU)
def vpop_trf_xlu0(state: ArchState, params: XLUSlotParams):
    """Pop the leading 8x8 window from the XLU buffer into a vector register.

    Reads the top 8x8 from the XLU buffer, writes to dest (left 8x8 of
    vreg), rolls the buffer up, and zeroes the vacated bottom rows.
    """
    width = max(state.num_sublanes, state.xlu_pop_width)
    width = min(width, state.xlu_buffer.shape[0], state.num_lanes)
    result = torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32)
    window = state.xlu_buffer[0:width, :].contiguous()
    if width == state.num_sublanes:
        result[:, 0:width] = window
    else:
        result[:, 0:width] = window.transpose(0, 1).contiguous()
    state.xlu_buffer = state.xlu_buffer.roll(-width, dims=0)
    state.xlu_buffer[-width:, :] = 0
    state.write_vreg(params.vd_reg, result)


# === MXU Instructions ===

@instr("vmatpush.msra.mxu0", BundleSlotType.MXU)
def vmatpush_msra_mxu0(state: ArchState, params: MXUSlotParams):
    """Push weight matrix (MSRA) from vector reg into MXU0.

    Loads the weight tile for matmul. MSRA = matrix storage / register file A.
    """
    src_vreg = params.vs1_reg
    state.push_mxu_weight("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu1", BundleSlotType.MXU)
def vmatpush_msra_mxu1(state: ArchState, params: MXUSlotParams):
    """Push weight matrix (MSRA) from vector reg into MXU1."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu2", BundleSlotType.MXU)
def vmatpush_msra_mxu2(state: ArchState, params: MXUSlotParams):
    """Push weight matrix (MSRA) from vector reg into MXU2."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu3", BundleSlotType.MXU)
def vmatpush_msra_mxu3(state: ArchState, params: MXUSlotParams):
    """Push weight matrix (MSRA) from vector reg into MXU3."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu0", BundleSlotType.MXU)
def vmatpush_xpose_msra_mxu0(state: ArchState, params: MXUSlotParams):
    """Push transposed weight matrix into MXU0.

    Same as vmatpush.msra but transposes the source before pushing;
    used for RHS-major layouts (e.g. transposed matmul).
    """
    src_vreg = params.vs1_reg
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu1", BundleSlotType.MXU)
def vmatpush_xpose_msra_mxu1(state: ArchState, params: MXUSlotParams):
    """Push transposed weight matrix into MXU1."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight_transpose("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu2", BundleSlotType.MXU)
def vmatpush_xpose_msra_mxu2(state: ArchState, params: MXUSlotParams):
    """Push transposed weight matrix into MXU2."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight_transpose("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu3", BundleSlotType.MXU)
def vmatpush_xpose_msra_mxu3(state: ArchState, params: MXUSlotParams):
    """Push transposed weight matrix into MXU3."""
    src_vreg = params.vs1_reg
    state.push_mxu_weight_transpose("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.bf16.xpose.msra.mxu0", BundleSlotType.MXU)
def vmatpush_bf16_xpose_msra_mxu0(state: ArchState, params: MXUSlotParams):
    """Push transposed BF16 weight matrix into MXU0.

    BF16 variant for mixed-precision matmul; source is BF16-packed.
    """
    src_vreg = params.vs1_reg
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.bfloat16))


@instr("vmatmul.f32.gmra.mxu0", BundleSlotType.MXU)
def vmatmul_f32_gmra_mxu0(state: ArchState, params: MXUSlotParams):
    """Matrix multiply: activation (GMRA) x weight (MSRA) in MXU0, accumulate.

    Reads activation from vsrc, multiplies with previously pushed weights,
    accumulates into the MXU accumulator. GMRA = general matrix register A.
    """
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.gmra.mxu1", BundleSlotType.MXU)
def vmatmul_f32_gmra_mxu1(state: ArchState, params: MXUSlotParams):
    """Matrix multiply: activation x weight in MXU1, accumulate."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.gmra.mxu2", BundleSlotType.MXU)
def vmatmul_f32_gmra_mxu2(state: ArchState, params: MXUSlotParams):
    """Matrix multiply: activation x weight in MXU2, accumulate."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.gmra.mxu3", BundleSlotType.MXU)
def vmatmul_f32_gmra_mxu3(state: ArchState, params: MXUSlotParams):
    """Matrix multiply: activation x weight in MXU3, accumulate."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu0", BundleSlotType.MXU)
def vmatmul_f32_vlgmr_msra_gmra_mxu0(state: ArchState, params: MXUSlotParams):
    """Matrix multiply in MXU0 (vlgmr/msra/gmra variant).

    Same matmul as vmatmul.f32.gmra; different operand routing/format.
    """
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu1", BundleSlotType.MXU)
def vmatmul_f32_vlgmr_msra_gmra_mxu1(state: ArchState, params: MXUSlotParams):
    """Matrix multiply in MXU1 (vlgmr/msra/gmra variant)."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu2", BundleSlotType.MXU)
def vmatmul_f32_vlgmr_msra_gmra_mxu2(state: ArchState, params: MXUSlotParams):
    """Matrix multiply in MXU2 (vlgmr/msra/gmra variant)."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu3", BundleSlotType.MXU)
def vmatmul_f32_vlgmr_msra_gmra_mxu3(state: ArchState, params: MXUSlotParams):
    """Matrix multiply in MXU3 (vlgmr/msra/gmra variant)."""
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.msk.f32.vlgmr.msra.gmra.mxu0", BundleSlotType.MXU)
def vmatmul_msk_f32_vlgmr_msra_gmra_mxu0(state: ArchState, params: MXUSlotParams):
    """Masked matrix multiply in MXU0.

    The first operand is a VM mask selecting active lanes in the activation
    tile; inactive lanes are zeroed before issuing matmul.
    """
    src_vreg = params.vs1_reg
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    vm_reg = params.vm_reg
    if vm_reg not in (0, "0", None):
        mask = _mask_operand(state, vm_reg)
        activation = torch.where(mask, activation, torch.zeros_like(activation))
    state.execute_mxu_matmul("mxu0", activation)


@instr("vpop.f32.mrf.mxu0", BundleSlotType.MXU)
def vpop_f32_mrf_mxu0(state: ArchState, params: MXUSlotParams):
    """Pop matrix result from MXU0 accumulator into vector register.

    MRF = matrix result/register file. Consumes one accumulated tile.
    """
    result = state.pop_mxu_accumulator("mxu0")
    state.write_vreg(params.vd_reg, result)


@instr("vpop.f32.mrf.mxu1", BundleSlotType.MXU)
def vpop_f32_mrf_mxu1(state: ArchState, params: MXUSlotParams):
    """Pop matrix result from MXU1 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu1")
    state.write_vreg(params.vd_reg, result)


@instr("vpop.f32.mrf.mxu2", BundleSlotType.MXU)
def vpop_f32_mrf_mxu2(state: ArchState, params: MXUSlotParams):
    """Pop matrix result from MXU2 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu2")
    state.write_vreg(params.vd_reg, result)


@instr("vpop.f32.mrf.mxu3", BundleSlotType.MXU)
def vpop_f32_mrf_mxu3(state: ArchState, params: MXUSlotParams):
    """Pop matrix result from MXU3 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu3")
    state.write_vreg(params.vd_reg, result)


@instr("vtrace", BundleSlotType.SALU)
def vtrace(state: ArchState, params: SALUSlotParams):
    """Tracing marker instruction (no-op)."""
    return


@instr("compiler-scheduling-barrier", BundleSlotType.SALU)
def compiler_scheduling_barrier(state: ArchState, params: SALUSlotParams):
    """Compiler scheduling fence (no-op)."""
    return


@instr("vsettm", BundleSlotType.SALU)
def vsettm(state: ArchState, params: SALUSlotParams):
    """Tile mask control (not modeled)."""
    return


@instr("setrngseed", BundleSlotType.SALU)
def setrngseed(state: ArchState, params: SALUSlotParams):
    """RNG seed setup (not modeled)."""
    return


@instr("vrng", BundleSlotType.SALU)
def vrng(state: ArchState, params: SALUSlotParams):
    """RNG value generation (returns zeros in this model)."""
    if params.vd_reg and params.vd_reg.startswith("v"):
        state.write_vreg(params.vd_reg, torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32))


@instr("vdelay", BundleSlotType.SALU)
def vdelay(state: ArchState, params: SALUSlotParams):
    """Pipeline delay slot (no-op)."""
    return


@instr("vsetiar.raw.iar0", BundleSlotType.SALU)
def vsetiar_raw_iar0(state: ArchState, params: SALUSlotParams):
    """IAR0 setup (not modeled)."""
    return


@instr("vsetiar.raw.iar1", BundleSlotType.SALU)
def vsetiar_raw_iar1(state: ArchState, params: SALUSlotParams):
    """IAR1 setup (not modeled)."""
    return
