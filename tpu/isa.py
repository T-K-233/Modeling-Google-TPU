from typing import Any

import torch

from .instruction import instr
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
    if isinstance(token, str) and token.startswith("s"):
        return state.read_xreg(token)
    return _parse_int(token)


def _as_u32(value: int) -> int:
    return value & U32_MASK


def _as_i32(value: int) -> int:
    value &= U32_MASK
    return value - (1 << 32) if value & (1 << 31) else value


def _predicate_active(state: ArchState, params: dict[str, Any]) -> bool:
    pred = params.get("pred")
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

@instr("inlined_call_operand.hbm")
def inlined_call_operand_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load HBM base address (in granule index, 16-byte units) for an inlined call operand.

    The parser populates args with the byte address; we store byte_addr // 16
    since subsequent sshll.u32 by 4 expects granule index.
    """
    byte_addr = params["imm1"]
    state.write_xreg(dest_reg, int(byte_addr))


@instr("inlined_call_operand.vmem")
def inlined_call_operand_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load VMEM base address (byte address) for an inlined call operand.

    Parser supplies the resolved byte address; we store it directly for
    use with vld/vst and scalar_lea.vmem.
    """
    byte_addr = params["imm1"]
    state.write_xreg(dest_reg, int(byte_addr))


@instr("inlined_call_operand.<no memory space>")
def inlined_call_operand_smem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load SMEM base address (in granule index, 16-byte units) for an inlined call operand.

    Same convention as inlined_call_operand.hbm: parser passes byte address,
    we store byte_addr // 16 for sshll.u32 by 4.
    """
    byte_addr = params["imm1"]
    state.write_xreg(dest_reg, int(byte_addr))


@instr("int_to_ptr.hbm")
def int_to_ptr_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Cast an integer to an HBM (high-bandwidth memory) pointer type.

    Reads the source register (last param) and writes it to the destination
    register. The source typically holds a byte address produced by
    sshll.u32 X, 4 (granule index * 16). This instruction annotates the
    value as an HBM pointer for type checking and for consumers such as
    dma.hbm_to_vmem, dma.vmem_to_hbm, and HBM-typed loads/stores.

    In the simulator model this is a pass-through: we copy the value
    since addresses are tracked as raw integers.
    """
    src_addr_reg = params["rs1"]
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(dest_reg, addr)


@instr("int_to_ptr.vmem")
def int_to_ptr_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Cast an integer to a VMEM (vector memory) pointer type.

    Reads the source register (last param) and writes it to the destination
    register. The source typically holds a byte address from sshll.u32 or
    scalar_lea.vmem. This annotates the value as a VMEM pointer for
    vld/vst and DMA operations.

    In the simulator this is a pass-through copy.
    """
    src_addr_reg = params["rs1"]
    addr = state.read_xreg(src_addr_reg)
    state.write_xreg(dest_reg, addr)


# === SFlag Instructions ===

@instr("vsyncpa")
def vsyncpa(state: ArchState, _: str, params: dict[str, Any]):
    """Store a scalar value into SFlag (synchronization / scalar flag memory).

    Writes the immediate or register value to the SFlag at the given byte
    address. Used for sync flags (e.g. DMA completion), loop counters, and
    other scalar state shared across lanes. Bounds-checked against sflag_size.
    """
    if not _predicate_active(state, params):
        return
    addr = params["addr"]
    value = params["rs1"]
    addr_val = _parse_operand(state, addr)
    assert (0 <= addr_val <= state.sflag_size - torch.uint32.itemsize), f"SFLAG address out of bounds: {addr_val}"
    state.write_sflag(addr_val, _parse_int(value))


@instr("vsyncadd")
def vsyncadd(state: ArchState, _: str, params: dict[str, Any]):
    """Atomic add into SFlag at the given byte address.

    Reads the current SFlag value, adds the immediate/register value (mod 256),
    and writes back. Used for reductions and lane coordination (e.g. computing
    per-lane offsets). Bounds-checked against sflag_size.
    """
    if not _predicate_active(state, params):
        return
    addr = params["addr"]
    value = params["rs1"]
    addr_val = _parse_operand(state, addr)
    if not (0 <= addr_val <= state.sflag_size - torch.uint32.itemsize):
        return
    flag_value = state.read_sflag(addr_val)
    flag_value = (flag_value + _parse_int(value)) % 256
    state.write_sflag(addr_val, flag_value)


# === DMA Transfer (HBM <-> VMEM) Instructions ===

@instr("dma.hbm_to_vmem")
def dma_hbm_to_vmem(state: ArchState, _: str, params: dict[str, Any]):
    """Copy data from HBM (host memory) into VMEM (vector memory).

    Takes source and destination addresses (from int_to_ptr.hbm / int_to_ptr.vmem),
    size in granules, and a sync flag address. Copies the block and sets the
    sync flag to 1. Addresses are in granule units (16-byte); size is scaled
    internally for the memory model.
    """
    if not _predicate_active(state, params):
        return
    src_addr_reg = params["rs1"]
    size_in_granules = params["imm1"]
    dest_addr_reg = params["rs2"]
    sync_flag = params["sync"]
    sync_flag_addr = _parse_operand(state, sync_flag)
    state.write_sflag(sync_flag_addr, 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = _parse_int(size_in_granules) << 5
    state.write_vmem(dest_addr, state.read_hbm(src_addr, size))


@instr("dma.vmem_to_hbm")
def dma_vmem_to_hbm(state: ArchState, _: str, params: dict[str, Any]):
    """Copy data from VMEM (vector memory) into HBM (host memory).

    Takes source and destination addresses, size in granules, and a sync flag.
    Copies the block and sets the sync flag to 1. Used for writing results
    back to host. Address/size units match dma.hbm_to_vmem.
    """
    if not _predicate_active(state, params):
        return
    src_addr_reg = params["rs1"]
    size_in_granules = params["imm1"]
    dest_addr_reg = params["rs2"]
    sync_flag = params["sync"]
    sync_flag_addr = _parse_operand(state, sync_flag)
    state.write_sflag(sync_flag_addr, 1)

    src_addr_granules = state.read_xreg(src_addr_reg)
    dest_addr_granules = state.read_xreg(dest_addr_reg)

    # TODO: not sure why need to divide by 16
    src_addr = src_addr_granules >> 4
    dest_addr = dest_addr_granules >> 4
    # TODO: not sure why need to multiply by 32
    size = _parse_int(size_in_granules) << 5
    state.write_hbm(dest_addr, state.read_vmem(src_addr, size))


@instr("dma.done.wait")
def dma_done_wait(state: ArchState, dest_reg: str, params: dict[str, Any]):
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

@instr("smov")
def smov(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar move: copy immediate or register value into a scalar register.

    One operand: move that value. Two operands with predicate: pred ? b : a
    (choose b if predicate true, else a). Writes result as u32.
    """
    value_a = params["rs1"]
    if params.get("pred") is None:
        value = _parse_operand(state, value_a)
    else:
        # TPU predicated scalar move form behaves as:
        #   smov (pred, a), b  => pred ? b : a
        active = _predicate_active(state, params)
        on_pred = value_a
        fallback = params["rs2"]
        chosen = fallback if active else on_pred
        value = _parse_operand(state, chosen)
    state.write_xreg(dest_reg, _as_u32(value))


# === SALU Instructions ===

@instr("sshll.u32")
def sshll_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar shift left logical: dest = src << imm (u32).

    Commonly used with imm=4 to convert granule index to byte address
    (multiply by 16) before int_to_ptr.hbm or int_to_ptr.vmem.
    """
    if not _predicate_active(state, params):
        return
    src_reg = params["rs1"]
    imm = params["imm1"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, src_reg) << _parse_int(imm)))


@instr("sshra.s32")
def sshra_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar shift right arithmetic: dest = src >> imm (s32, sign-extended).

    Used for pointer arithmetic or dividing signed values by powers of two.
    """
    if not _predicate_active(state, params):
        return
    src_reg = params["rs1"]
    imm = params["imm1"]
    value = _as_i32(_parse_operand(state, src_reg))
    state.write_xreg(dest_reg, _as_u32(value >> _parse_int(imm)))


@instr("sadd.s32")
def sadd_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar add: dest = a + b (s32, truncated to u32)."""
    if not _predicate_active(state, params):
        return
    a = params["rs1"]
    b = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) + _parse_operand(state, b)))


@instr("ssub.s32")
def ssub_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar subtract: dest = a - b (s32, truncated to u32)."""
    if not _predicate_active(state, params):
        return
    a = params["rs1"]
    b = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) - _parse_operand(state, b)))


@instr("sor.u32")
def sor_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar bitwise or: dest = a | b (u32)."""
    if not _predicate_active(state, params):
        return
    a = params["rs1"]
    b = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) | _parse_operand(state, b)))


@instr("sand.u32")
def sand_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar bitwise and: dest = a & b (u32)."""
    if not _predicate_active(state, params):
        return
    a = params["rs1"]
    b = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) & _parse_operand(state, b)))


@instr("scalar_select")
def scalar_select(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Select scalar by predicate: dest = pred ? on_true : on_false.

    Reads the predicate register and writes the chosen scalar value.
    """
    pred_reg = params["pred"]
    on_true = params["rs1"]
    on_false = params["rs2"]
    chosen = on_true if state.read_preg(pred_reg) else on_false
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, chosen)))


@instr("sphi")
def sphi(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar phi (SSA merge): pick value based on control flow.

    In SSA form this would merge values from different predecessors. After
    register coalescing in the compiler, it reduces to a move.
    """
    src = params["rs1"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, src)))


@instr("scmp.eq.s32.totalorder")
def scmp_eq_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare equal: dest_pred = (a == b) as s32."""
    a = params["rs1"]
    b = params["rs2"]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) == _as_i32(_parse_operand(state, b)))


@instr("scmp.ne.s32.totalorder")
def scmp_ne_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare not-equal: dest_pred = (a != b) as s32."""
    a = params["rs1"]
    b = params["rs2"]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) != _as_i32(_parse_operand(state, b)))


@instr("scmp.ge.s32.totalorder")
def scmp_ge_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare greater-or-equal: dest_pred = (a >= b) as s32."""
    a = params["rs1"]
    b = params["rs2"]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) >= _as_i32(_parse_operand(state, b)))


@instr("scmp.lt.s32.totalorder")
def scmp_lt_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare less-than: dest_pred = (a < b) as s32."""
    a = params["rs1"]
    b = params["rs2"]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) < _as_i32(_parse_operand(state, b)))


@instr("por")
def por(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate or: dest = p0 or p1."""
    p0 = params["ps1"]
    p1 = params["ps2"]
    state.write_preg(dest_reg, state.read_preg(p0) or state.read_preg(p1))


@instr("pnand")
def pnand(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate nand: dest = p0 and (not p1)."""
    p0 = params["ps1"]
    p1 = params["ps2"]
    state.write_preg(dest_reg, state.read_preg(p0) and (not state.read_preg(p1)))


@instr("pneg")
def pneg(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate negate: dest = not p0."""
    p0 = params["ps1"]
    state.write_preg(dest_reg, not state.read_preg(p0))


@instr("scalar_lea.vmem")
def scalar_lea_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for VMEM: dest = base + (offset << 9).

    Computes VMEM byte address from base and scaled offset. Used for
    indexing into vector memory with tile or row strides.
    """
    if not _predicate_active(state, params):
        return
    base = params["rs1"]
    offset = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.hbm")
def scalar_lea_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for HBM: dest = base + (offset << 9).

    Computes HBM byte address from base and scaled offset.
    """
    if not _predicate_active(state, params):
        return
    base = params["rs1"]
    offset = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.sflag")
def scalar_lea_sflag(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for SFlag: dest = base + (offset << 2).

    Offset scaled by 4 (uint32 size) for indexing SFlag entries.
    """
    if not _predicate_active(state, params):
        return
    base = params["rs1"]
    offset = params["rs2"]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 2)))


@instr("sst")
def sst(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar store: write 4-byte value from src_reg to SMEM at smem_addr.

    SMEM address is in bytes. Little-endian.
    """
    smem_addr = params["addr"]
    src_reg = params["rs1"]
    address = _parse_operand(state, smem_addr)
    value = _as_u32(_parse_operand(state, src_reg))
    raw = torch.tensor(list(value.to_bytes(4, byteorder="little", signed=False)), dtype=torch.uint8)
    state.write_smem(address, raw)


@instr("sld")
def sld(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar load: read 4 bytes from SMEM at smem_addr into dest_reg.

    Little-endian. Used for loading constants or scalar values from SMEM.
    """
    smem_addr = params["addr"]
    address = _parse_operand(state, smem_addr)
    raw = state.read_smem(address, 4, dtype=torch.uint8).tolist()
    state.write_xreg(dest_reg, int.from_bytes(bytes(raw), byteorder="little", signed=False))


@instr("sbr.rel")
def sbr_rel(state: ArchState, _: str, params: dict[str, Any]):
    """Relative scalar branch: set PC to target (bundle index)."""
    if not _predicate_active(state, params):
        return
    state.next_pc = _parse_int(params["target"])


@instr("shalt.err")
def shalt_err(state: ArchState, _: str, params: dict[str, Any]):
    """Halt on error: predicate-guarded trap for bounds checks or assertions.

    On hardware would halt the core. In this functional simulator we treat
    it as non-fatal (no-op) for continued execution.
    """
    if not _predicate_active(state, params):
        return
    # Bounds checks are modeled as non-fatal in this functional simulator.
    return


# === Tensor Memory Load/Store Instructions ===

@instr("vstv")
def vstv(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load and broadcast a scalar from SMEM into a vector register.

    Reads 4 bytes (float32) at the address in src_reg and replicates it
    across all sublanes and lanes. Used for broadcasting constants into
    vector ops (e.g. before vpack for BF16 conversion).
    """
    src_reg = params["rs1"]
    address = state.read_xreg(src_reg)
    scalar_data = state.read_smem(address, 4, dtype=torch.float32)
    data = torch.tensor([scalar_data], dtype=torch.float32).repeat(state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, data)


@instr("vld")
def vld(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector load from VMEM into a vector register.

    Address can be register, register+offset, or immediate. Optional
    sublane mask (8-bit) zeroes out masked rows. Loads vreg_size bytes
    and reshapes to num_sublanes x lanes.
    """
    sreg_or_imm = str(params["addr"])
    sublane_mask = str(params.get("sm", "255"))
    ss_stride = None
    ss_token = params.get("ss", 0)
    if isinstance(ss_token, str) and ss_token.startswith("ss="):
        ss_stride = int(ss_token.split("=", 1)[1])

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

    # ss:$0 means all sublanes read from the same base row.
    if ss_stride == 0:
        data = data[0:1, :].repeat(state.num_sublanes, 1)

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
    """Vector store from a vector register to VMEM.

    Address (register or immediate), optional sublane mask (8-bit,
    bit i = 1 stores row i), and source register. Mask 0 stores nothing.
    """
    address = params["addr"]
    sublane_mask = params["sm"]
    vsrc_reg = params["vs1"]
    data = state.read_vreg(vsrc_reg, dtype=torch.uint8)
    if isinstance(address, str) and address.startswith("s"):
        address = state.read_xreg(address)
    else:
        address_str = str(address)
        address = int(address_str, 16) if address_str.startswith("0x") else int(address_str)

    mask_val = int(sublane_mask)
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


@instr("vst.msk")
def vst_msk(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector store to VMEM with sublane mask (dest_reg variant).

    Same as vst but with a dedicated mask form; bit i = 1 stores row i.
    """
    address = params["addr"]
    sublane_mask = params["sm"]
    vsrc_reg = params["vs1"]
    data = state.read_vreg(vsrc_reg, dtype=torch.uint8)
    if isinstance(address, str) and address.startswith("s"):
        address = state.read_xreg(address)
    else:
        address_str = str(address)
        address = int(address_str, 16) if address_str.startswith("0x") else int(address_str)

    mask_val = int(sublane_mask)
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

@instr("vadd.f32")
def vadd_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector add (float32): dest = vsrc1 + vsrc2 elementwise.

    Operands can be vector registers or immediates.
    """
    vsrc1_reg = params["vs1"]
    vsrc2_reg = params["vs2"]
    if isinstance(vsrc1_reg, str) and vsrc1_reg.startswith("v"):
        vsrc1_data = state.read_vreg(vsrc1_reg, dtype=torch.float32)
    else:
        vsrc1_data = float(vsrc1_reg)
    if isinstance(vsrc2_reg, str) and vsrc2_reg.startswith("v"):
        vsrc2_data = state.read_vreg(vsrc2_reg, dtype=torch.float32)
    else:
        vsrc2_data = float(vsrc2_reg)

    result = vsrc1_data + vsrc2_data
    state.write_vreg(dest_reg, result)


@instr("vmov")
def vmov(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Broadcast scalar immediate/register value into a vector register."""
    src = params["vs1"]
    if isinstance(src, str) and (src.startswith("v") or src in state.vreg):
        state.write_vreg(dest_reg, state.read_vreg(src, dtype=torch.float32))
        return
    if _is_float_token(src):
        state.write_vreg(dest_reg, _full_f32(state, _parse_float(src)))
        return
    state.write_vreg(dest_reg, _full_u32(state, _parse_int(src)))


@instr("vadd.s32")
def vadd_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector add in 32-bit integer lanes with wraparound."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(dest_reg, ((lhs + rhs) & U32_MASK).to(torch.uint32))


@instr("vsub.s32")
def vsub_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector subtract in 32-bit integer lanes with wraparound."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(dest_reg, ((lhs - rhs) & U32_MASK).to(torch.uint32))


@instr("vsub.f32")
def vsub_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector float subtraction."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vreg(dest_reg, _vector_operand_f32(state, a) - _vector_operand_f32(state, b))


@instr("vmul.f32")
def vmul_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector float multiplication."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vreg(dest_reg, _vector_operand_f32(state, a) * _vector_operand_f32(state, b))


@instr("vmul.u32")
def vmul_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector 32-bit unsigned multiply with wraparound."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vreg(dest_reg, ((lhs * rhs) & U32_MASK).to(torch.uint32))


@instr("vand.u32")
def vand_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector bitwise AND on 32-bit lanes."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vreg(dest_reg, _vector_operand_u32(state, a) & _vector_operand_u32(state, b))


@instr("vor.u32")
def vor_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector bitwise OR on 32-bit lanes."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vreg(dest_reg, _vector_operand_u32(state, a) | _vector_operand_u32(state, b))


@instr("vxor.u32")
def vxor_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector bitwise XOR on 32-bit lanes."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vreg(dest_reg, _vector_operand_u32(state, a) ^ _vector_operand_u32(state, b))


@instr("vshll.u32")
def vshll_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector logical left shift on 32-bit lanes."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    sh = (_vector_operand_u32(state, b) & 31).to(torch.int64)
    state.write_vreg(dest_reg, ((lhs << sh) & U32_MASK).to(torch.uint32))


@instr("vshrl.u32")
def vshrl_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector logical right shift on 32-bit lanes."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    sh = (_vector_operand_u32(state, b) & 31).to(torch.int64)
    state.write_vreg(dest_reg, ((lhs >> sh) & U32_MASK).to(torch.uint32))


@instr("vclz")
def vclz(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector count-leading-zeros for 32-bit lanes."""
    src = params["vs1"]
    values = _vector_operand_u32(state, src).flatten().tolist()
    clz = [
        (32 - int(v).bit_length()) if int(v) != 0 else 32
        for v in values
    ]
    result = torch.tensor(clz, dtype=torch.uint32).reshape(state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, result)


@instr("vcvt.s32.f32")
def vcvt_s32_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Convert vector int32 lanes to float32 lanes."""
    src = params["vs1"]
    state.write_vreg(dest_reg, _vector_operand_i32(state, src).to(torch.float32))


@instr("vcmp.lt.s32.totalorder")
def vcmp_lt_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector signed int compare (<), producing VM mask."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vmreg(dest_reg, _vector_operand_i32(state, a) < _vector_operand_i32(state, b))


@instr("vcmp.gt.s32.totalorder")
def vcmp_gt_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector signed int compare (>), producing VM mask."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vmreg(dest_reg, _vector_operand_i32(state, a) > _vector_operand_i32(state, b))


@instr("vcmp.eq.s32.totalorder")
def vcmp_eq_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector signed int compare (==), producing VM mask."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vmreg(dest_reg, _vector_operand_i32(state, a) == _vector_operand_i32(state, b))


@instr("vcmp.le.f32.partialorder")
def vcmp_le_f32_partialorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector float compare (<=) with partial-order NaN handling."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs <= rhs)
    state.write_vmreg(dest_reg, mask)


@instr("vcmp.eq.f32.partialorder")
def vcmp_eq_f32_partialorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector float compare (==) with partial-order NaN handling."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_f32(state, a)
    rhs = _vector_operand_f32(state, b)
    mask = torch.isfinite(lhs) & torch.isfinite(rhs) & (lhs == rhs)
    state.write_vmreg(dest_reg, mask)


@instr("vc.u32")
def vc_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector carry-out mask for 32-bit unsigned addition."""
    a = params["vs1"]
    b = params["vs2"]
    lhs = _vector_operand_u32(state, a).to(torch.int64)
    rhs = _vector_operand_u32(state, b).to(torch.int64)
    state.write_vmreg(dest_reg, (lhs + rhs) > U32_MASK)


@instr("vmor")
def vmor(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Mask OR."""
    a = params["vs1"]
    b = params["vs2"]
    state.write_vmreg(dest_reg, _mask_operand(state, a) | _mask_operand(state, b))


@instr("vsel")
def vsel(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector select by VM mask: mask ? on_true : on_false."""
    vm_reg = params["vm1"]
    on_true = params["vs1"]
    on_false = params["vs2"]
    mask = _mask_operand(state, vm_reg)
    true_bits = _vector_operand_u32(state, on_true)
    false_bits = _vector_operand_u32(state, on_false)
    selected = torch.where(mask, true_bits.to(torch.int64), false_bits.to(torch.int64)).to(torch.uint32)
    state.write_vreg(dest_reg, selected)


@instr("vlaneseq")
def vlaneseq(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Lane index sequence [0..127] replicated across sublanes."""
    seq = torch.arange(state.num_lanes, dtype=torch.int64).to(torch.uint32).unsqueeze(0).repeat(state.num_sublanes, 1)
    state.write_vreg(dest_reg, seq)


@instr("vset.pattern.permute.xlu0")
def vset_pattern_permute_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Record permute pattern token (functional model)."""
    # Pattern metadata is not fully modeled; keep the latest token around so
    # vperm/vpop.permute can coordinate temporary ids.
    if dest_reg:
        state.last_permute_token = dest_reg


@instr("vperm.xlu0")
def vperm_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Queue a source vector for a subsequent vpop.permute.xlu0."""
    source = None
    for token in reversed(list(params.values())):
        if isinstance(token, str) and token.startswith("v"):
            source = token
            break
    if source is None:
        return
    token = dest_reg if dest_reg else f"perm_{len(state.permute_buffer)}"
    state.permute_buffer[token] = state.read_vreg(source, dtype=torch.float32).clone()
    state.last_permute_token = token


@instr("vpop.permute.xlu0")
def vpop_permute_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop permuted data from the functional permute buffer.

    Current model covers the emitted TPU patterns used by these kernels:
    source vectors of logical shape [8] are expanded into per-row broadcasts.
    """
    token = params.get("imm1", state.last_permute_token)
    if token is None or (isinstance(token, int) and token == 0):
        token = state.last_permute_token
    if token is None or token not in state.permute_buffer:
        state.write_vreg(dest_reg, torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32))
        return
    src = state.permute_buffer[token]
    vec8 = src[0, :state.num_sublanes]
    out = vec8.unsqueeze(1).repeat(1, state.num_lanes).contiguous()
    state.write_vreg(dest_reg, out)


@instr("vrcp.f32")
def vrcp_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector reciprocal approximation (modeled as exact reciprocal)."""
    src = params["vs1"]
    state.write_vreg(dest_reg, 1.0 / _vector_operand_f32(state, src))


@instr("vpow2.f32")
def vpow2_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector base-2 exponent."""
    src = params["vs1"]
    state.write_vreg(dest_reg, torch.pow(2.0, _vector_operand_f32(state, src)))


@instr("vpop.eup")
def vpop_eup(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop EUP pipeline value (modeled as identity)."""
    src = params["vs1"]
    state.write_vreg(dest_reg, _vector_operand_f32(state, src))


@instr("vweird.f32")
def vweird_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Special-value detector for transcendental pipelines."""
    src = params["vs1"]
    data = _vector_operand_f32(state, src)
    state.write_vmreg(dest_reg, ~torch.isfinite(data))


@instr("vtanh.f32")
def vtanh_f32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Vector tanh."""
    src = params["vs1"]
    state.write_vreg(dest_reg, torch.tanh(_vector_operand_f32(state, src)))


@instr("vrot.slane")
def vrot_slane(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Rotate vector rows (sublanes) by shift_amount positions.

    Rolls the sublane dimension; used for data movement in reductions
    and lane communication.
    """
    vsrc_reg = params["vs1"]
    shift_amount = params["imm1"]
    shift_amount = int(shift_amount)
    assert shift_amount >= 0 and shift_amount < state.num_sublanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    vdest_data = vsrc_data.roll(shift_amount, dims=0)
    state.write_vreg(dest_reg, vdest_data)


@instr("vcmask")
def vcmask(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Configure vector mask (submatrix selection for loads/stores).

    Vector mask registers are not explicitly modeled in this simulator;
    the instruction is a no-op.
    """
    return


# === Tensor Packing/Unpacking Instructions ===

@instr("vpack.c.bf16")
def vpack_c_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pack two float32 registers (high, low) into one BF16 register.

    Concatenates high and low as BF16; typically used after vadd/vstv
    to prepare data for MXU matmul.
    """
    reg1_or_imm = params["vs1"]
    vsrc2_reg = params["vs2"]
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
    state.write_vreg(dest_reg, packed)


@instr("vunpack.c.l.bf16")
def vunpack_c_l_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Unpack low half of packed BF16 register to float32.

    Extracts the lower 16 bits of each 32-bit lane; used after MXU
    output to convert BF16 results back to FP32.
    """
    vsrc_reg = params["vs1"]
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    low, _ = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, low.to(torch.float32))


@instr("vunpack.c.h.bf16")
def vunpack_c_h_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Unpack high half of packed BF16 register to float32.

    Extracts the upper 16 bits of each 32-bit lane.
    """
    vsrc_reg = params["vs1"]
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    _, high = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, high.to(torch.float32))


@instr("vunpack.i.l.bf16")
def vunpack_i_l_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Unpack low half (immediate path): reinterpret FP32 lanes as BF16 low.

    For values from scalar path (e.g. vstv) represented as FP32; converts
    to BF16 and back to FP32 for downstream use.
    """
    vsrc_reg = params["vs1"]
    # Immediate BF16 values originate from scalar paths (e.g. vstv), where values
    # are represented in FP32 lanes and then interpreted as BF16 for conversion.
    unpacked_data = state.read_vreg(vsrc_reg, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    state.write_vreg(dest_reg, unpacked_data.contiguous())


# === XLU Instructions ===


@instr("vxpose.xlu0.b32.start.end")
def vxpose_xlu0_b32_start_end(state: ArchState, _: str, params: dict[str, Any]):
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
    vsrc_reg = params["vs1"]
    num_lanes = params["imm1"]
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[0:num_lanes, :] = vsrc_data.transpose(0, 1)
    state.xlu_pop_width = state.num_sublanes


@instr("vxpose.xlu0.b32.start")
def vxpose_xlu0_b32_start(state: ArchState, _: str, params: dict[str, Any]):
    """Transpose and load into XLU buffer (start of sequence).

    Transposes the source tensor (num_lanes x sublane) and stores it in
    the top rows of the XLU buffer. Used with vxpose.xlu0.b32.end for
    multi-tile transpose.
    """
    vsrc_reg = params["vs1"]
    num_lanes = params["imm1"]
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[0:num_lanes, :] = vsrc_data.transpose(0, 1)
    state.xlu_pop_width = state.num_sublanes


@instr("vxpose.xlu0.b32.end")
def vxpose_xlu0_b32_end(state: ArchState, _: str, params: dict[str, Any]):
    """Transpose and load into XLU buffer (end of sequence).

    Rolls the buffer up, then writes transposed source into the next
    rows. Complements vxpose.xlu0.b32.start for sliding-window transpose.
    """
    vsrc_reg = params["vs1"]
    num_lanes = params["imm1"]
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    # For a start/end pair, append into the second half of the XLU staging
    # buffer so subsequent vpop.trf can consume both halves sequentially.
    state.xlu_buffer[state.num_lanes:state.num_lanes + num_lanes, :] = vsrc_data.transpose(0, 1)
    state.xlu_pop_width = state.num_sublanes * 2


@instr("vpop.trf.xlu0")
def vpop_trf_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop the leading 8x8 window from the XLU buffer into a vector register.

    Reads the top 8x8 from the XLU buffer, writes to dest (left 8x8 of
    vreg), rolls the buffer up, and zeroes the vacated bottom rows.
    """
    width = max(state.num_sublanes, int(state.xlu_pop_width))
    width = min(width, state.xlu_buffer.shape[0], state.num_lanes)
    result = torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32)
    window = state.xlu_buffer[0:width, :].contiguous()
    if width == state.num_sublanes:
        result[:, 0:width] = window
    else:
        result[:, 0:width] = window.transpose(0, 1).contiguous()
    state.xlu_buffer = state.xlu_buffer.roll(-width, dims=0)
    state.xlu_buffer[-width:, :] = 0
    state.write_vreg(dest_reg, result)


# === MXU Instructions ===

@instr("vmatpush.msra.mxu0")
def vmatpush_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU0.

    Loads the weight tile for matmul. MSRA = matrix storage / register file A.
    """
    src_vreg = params["vs1"]
    state.push_mxu_weight("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu1")
def vmatpush_msra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU1."""
    src_vreg = params["vs1"]
    state.push_mxu_weight("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu2")
def vmatpush_msra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU2."""
    src_vreg = params["vs1"]
    state.push_mxu_weight("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu3")
def vmatpush_msra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU3."""
    src_vreg = params["vs1"]
    state.push_mxu_weight("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu0")
def vmatpush_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU0.

    Same as vmatpush.msra but transposes the source before pushing;
    used for RHS-major layouts (e.g. transposed matmul).
    """
    src_vreg = params["vs1"]
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu1")
def vmatpush_xpose_msra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU1."""
    src_vreg = params["vs1"]
    state.push_mxu_weight_transpose("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu2")
def vmatpush_xpose_msra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU2."""
    src_vreg = params["vs1"]
    state.push_mxu_weight_transpose("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu3")
def vmatpush_xpose_msra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU3."""
    src_vreg = params["vs1"]
    state.push_mxu_weight_transpose("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.bf16.xpose.msra.mxu0")
def vmatpush_bf16_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed BF16 weight matrix into MXU0.

    BF16 variant for mixed-precision matmul; source is BF16-packed.
    """
    src_vreg = params["vs1"]
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.bfloat16))


@instr("vmatmul.f32.gmra.mxu0")
def vmatmul_f32_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation (GMRA) x weight (MSRA) in MXU0, accumulate.

    Reads activation from vsrc, multiplies with previously pushed weights,
    accumulates into the MXU accumulator. GMRA = general matrix register A.
    """
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.gmra.mxu1")
def vmatmul_f32_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU1, accumulate."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.gmra.mxu2")
def vmatmul_f32_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU2, accumulate."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.gmra.mxu3")
def vmatmul_f32_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU3, accumulate."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu0")
def vmatmul_f32_vlgmr_msra_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU0 (vlgmr/msra/gmra variant).

    Same matmul as vmatmul.f32.gmra; different operand routing/format.
    """
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu1")
def vmatmul_f32_vlgmr_msra_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU1 (vlgmr/msra/gmra variant)."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu2")
def vmatmul_f32_vlgmr_msra_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU2 (vlgmr/msra/gmra variant)."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu3")
def vmatmul_f32_vlgmr_msra_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU3 (vlgmr/msra/gmra variant)."""
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.msk.f32.vlgmr.msra.gmra.mxu0")
def vmatmul_msk_f32_vlgmr_msra_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Masked matrix multiply in MXU0.

    The first operand is a VM mask selecting active lanes in the activation
    tile; inactive lanes are zeroed before issuing matmul.
    """
    src_vreg = params["vs1"]
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    vm_reg = params.get("vm1")
    if vm_reg not in (0, "0", None):
        mask = _mask_operand(state, vm_reg)
        activation = torch.where(mask, activation, torch.zeros_like(activation))
    state.execute_mxu_matmul("mxu0", activation)


@instr("vpop.f32.mrf.mxu0")
def vpop_f32_mrf_mxu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop matrix result from MXU0 accumulator into vector register.

    MRF = matrix result/register file. Consumes one accumulated tile.
    """
    result = state.pop_mxu_accumulator("mxu0")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu1")
def vpop_f32_mrf_mxu1(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop matrix result from MXU1 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu1")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu2")
def vpop_f32_mrf_mxu2(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop matrix result from MXU2 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu2")
    state.write_vreg(dest_reg, result)


@instr("vpop.f32.mrf.mxu3")
def vpop_f32_mrf_mxu3(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop matrix result from MXU3 accumulator into vector register."""
    result = state.pop_mxu_accumulator("mxu3")
    state.write_vreg(dest_reg, result)
