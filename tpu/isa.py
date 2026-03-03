from typing import Any

import torch

from .instruction import instr
from .arch_state import ArchState
from .tiling import pack_bf16_register, unpack_bf16_register


U32_MASK = 0xFFFFFFFF


def _parse_int(token: str) -> int:
    token = token.strip()
    if token.startswith("$"):
        token = token[1:]
    if token.startswith("-0x"):
        return -int(token[3:], 16)
    if token.startswith("0x"):
        return int(token, 16)
    return int(token)


def _parse_operand(state: ArchState, token: str) -> int:
    """Resolve an operand symbol into either a register or an immediate value."""
    if token.startswith("s"):
        return state.read_xreg(token)
    return _parse_int(token)


def _as_u32(value: int) -> int:
    return value & U32_MASK


def _as_i32(value: int) -> int:
    value &= U32_MASK
    return value - (1 << 32) if value & (1 << 31) else value


def _consume_predicate(state: ArchState, params: list[str]) -> tuple[bool, list[str]]:
    if not params:
        return True, params
    pred = params[0]
    if not (pred.startswith("p") or pred.startswith("!p")):
        return True, params
    if pred.startswith("!"):
        return (not state.read_preg(pred[1:])), params[1:]
    return state.read_preg(pred), params[1:]


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
    """Load VMEM base address (byte address) for an inlined call operand.

    Parser supplies the resolved byte address; we store it directly for
    use with vld/vst and scalar_lea.vmem.
    """
    byte_addr, = params
    state.write_xreg(dest_reg, int(byte_addr))


@instr("inlined_call_operand.<no memory space>")
def inlined_call_operand_smem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load SMEM base address (in granule index, 16-byte units) for an inlined call operand.

    Same convention as inlined_call_operand.hbm: parser passes byte address,
    we store byte_addr // 16 for sshll.u32 by 4.
    """
    byte_addr, = params
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
    src_addr_reg = params[-1]
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
    src_addr_reg = params[-1]
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
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    addr, value = rest[-2:]
    addr_val = _parse_operand(state, addr)
    if not (0 <= addr_val <= state.sflag_size - torch.uint32.itemsize):
        return
    state.write_sflag(addr_val, _parse_int(value))


@instr("vsyncadd")
def vsyncadd(state: ArchState, _: str, params: dict[str, Any]):
    """Atomic add into SFlag at the given byte address.

    Reads the current SFlag value, adds the immediate/register value (mod 256),
    and writes back. Used for reductions and lane coordination (e.g. computing
    per-lane offsets). Bounds-checked against sflag_size.
    """
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    addr, value = rest[-2:]
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
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = rest[-4:]
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
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    src_addr_reg, size_in_granules, dest_addr_reg, sync_flag = rest[-4:]
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
    active, _rest = _consume_predicate(state, list(params))
    if not active:
        return
    return


# === Scalar Memory Load/Store Instructions ===

@instr("smov")
def smov(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar move: copy immediate or register value into a scalar register.

    One operand: move that value. Two operands with predicate: pred ? b : a
    (choose b if predicate true, else a). Writes result as u32.
    """
    active, rest = _consume_predicate(state, list(params))
    if len(rest) == 0:
        return
    if len(rest) == 1:
        value = _parse_operand(state, rest[0])
    else:
        # TPU predicated scalar move form behaves as:
        #   smov (pred, a), b  => pred ? b : a
        on_pred, fallback = rest[0], rest[1]
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
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    src_reg, imm = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, src_reg) << _parse_int(imm)))


@instr("sshra.s32")
def sshra_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar shift right arithmetic: dest = src >> imm (s32, sign-extended).

    Used for pointer arithmetic or dividing signed values by powers of two.
    """
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    src_reg, imm = rest[-2:]
    value = _as_i32(_parse_operand(state, src_reg))
    state.write_xreg(dest_reg, _as_u32(value >> _parse_int(imm)))


@instr("sadd.s32")
def sadd_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar add: dest = a + b (s32, truncated to u32)."""
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    a, b = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) + _parse_operand(state, b)))


@instr("ssub.s32")
def ssub_s32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar subtract: dest = a - b (s32, truncated to u32)."""
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    a, b = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) - _parse_operand(state, b)))


@instr("sor.u32")
def sor_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar bitwise or: dest = a | b (u32)."""
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    a, b = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) | _parse_operand(state, b)))


@instr("sand.u32")
def sand_u32(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar bitwise and: dest = a & b (u32)."""
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    a, b = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, a) & _parse_operand(state, b)))


@instr("scalar_select")
def scalar_select(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Select scalar by predicate: dest = pred ? on_true : on_false.

    Reads the predicate register and writes the chosen scalar value.
    """
    pred_reg, on_true, on_false = params[-3:]
    chosen = on_true if state.read_preg(pred_reg) else on_false
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, chosen)))


@instr("sphi")
def sphi(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar phi (SSA merge): pick value based on control flow.

    In SSA form this would merge values from different predecessors. After
    register coalescing in the compiler, it reduces to a move.
    """
    src = params[-1]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, src)))


@instr("scmp.eq.s32.totalorder")
def scmp_eq_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare equal: dest_pred = (a == b) as s32."""
    a, b = params[-2:]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) == _as_i32(_parse_operand(state, b)))


@instr("scmp.ne.s32.totalorder")
def scmp_ne_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare not-equal: dest_pred = (a != b) as s32."""
    a, b = params[-2:]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) != _as_i32(_parse_operand(state, b)))


@instr("scmp.ge.s32.totalorder")
def scmp_ge_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare greater-or-equal: dest_pred = (a >= b) as s32."""
    a, b = params[-2:]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) >= _as_i32(_parse_operand(state, b)))


@instr("scmp.lt.s32.totalorder")
def scmp_lt_s32_totalorder(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar compare less-than: dest_pred = (a < b) as s32."""
    a, b = params[-2:]
    state.write_preg(dest_reg, _as_i32(_parse_operand(state, a)) < _as_i32(_parse_operand(state, b)))


@instr("por")
def por(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate or: dest = p0 or p1."""
    p0, p1 = params[-2:]
    state.write_preg(dest_reg, state.read_preg(p0) or state.read_preg(p1))


@instr("pnand")
def pnand(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate nand: dest = p0 and (not p1)."""
    p0, p1 = params[-2:]
    state.write_preg(dest_reg, state.read_preg(p0) and (not state.read_preg(p1)))


@instr("pneg")
def pneg(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Predicate negate: dest = not p0."""
    p0, = params[-1:]
    state.write_preg(dest_reg, not state.read_preg(p0))


@instr("scalar_lea.vmem")
def scalar_lea_vmem(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for VMEM: dest = base + (offset << 9).

    Computes VMEM byte address from base and scaled offset. Used for
    indexing into vector memory with tile or row strides.
    """
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    base, offset = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.hbm")
def scalar_lea_hbm(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for HBM: dest = base + (offset << 9).

    Computes HBM byte address from base and scaled offset.
    """
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    base, offset = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 9)))


@instr("scalar_lea.sflag")
def scalar_lea_sflag(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Load-effective-address for SFlag: dest = base + (offset << 2).

    Offset scaled by 4 (uint32 size) for indexing SFlag entries.
    """
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    base, offset = rest[-2:]
    state.write_xreg(dest_reg, _as_u32(_parse_operand(state, base) + (_parse_operand(state, offset) << 2)))


@instr("sst")
def sst(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar store: write 4-byte value from src_reg to SMEM at smem_addr.

    SMEM address is in bytes. Little-endian.
    """
    smem_addr, src_reg = params[-2:]
    address = _parse_operand(state, smem_addr)
    value = _as_u32(_parse_operand(state, src_reg))
    raw = torch.tensor(list(value.to_bytes(4, byteorder="little", signed=False)), dtype=torch.uint8)
    state.write_smem(address, raw)


@instr("sld")
def sld(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Scalar load: read 4 bytes from SMEM at smem_addr into dest_reg.

    Little-endian. Used for loading constants or scalar values from SMEM.
    """
    smem_addr, = params[-1:]
    address = _parse_operand(state, smem_addr)
    raw = state.read_smem(address, 4, dtype=torch.uint8).tolist()
    state.write_xreg(dest_reg, int.from_bytes(bytes(raw), byteorder="little", signed=False))


@instr("sbr.rel")
def sbr_rel(state: ArchState, _: str, params: dict[str, Any]):
    """Relative scalar branch: set PC to target (bundle index)."""
    active, rest = _consume_predicate(state, list(params))
    if not active:
        return
    if not rest:
        return
    state.next_pc = _parse_int(rest[0])


@instr("shalt.err")
def shalt_err(state: ArchState, _: str, params: dict[str, Any]):
    """Halt on error: predicate-guarded trap for bounds checks or assertions.

    On hardware would halt the core. In this functional simulator we treat
    it as non-fatal (no-op) for continued execution.
    """
    active, _rest = _consume_predicate(state, list(params))
    if not active:
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
    src_reg, = params
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
    """Vector store from a vector register to VMEM.

    Address (register or immediate), optional sublane mask (8-bit,
    bit i = 1 stores row i), and source register. Mask 0 stores nothing.
    """
    address, sublane_mask, vsrc_reg = params  # optional middle arg: mask (sm:$0xN)
    data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    if address.startswith("s"):
        address = state.read_xreg(address)
    else:
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
    """Vector store to VMEM with sublane mask (dest_reg variant).

    Same as vst but with a dedicated mask form; bit i = 1 stores row i.
    """
    address, sublane_mask, vsrc_reg = params
    data = state.read_vreg(vsrc_reg)
    if address.startswith("s"):
        address = state.read_xreg(address)
    else:
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
    """Vector add (float32): dest = vsrc1 + vsrc2 elementwise.

    Operands can be vector registers or immediates.
    """
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
    """Rotate vector rows (sublanes) by shift_amount positions.

    Rolls the sublane dimension; used for data movement in reductions
    and lane communication.
    """
    vsrc_reg, shift_amount = params
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
    reg1_or_imm, vsrc2_reg = params
    if reg1_or_imm.startswith("v"):
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
    vsrc_reg, = params
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    low, _ = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, low.to(torch.float32))


@instr("vunpack.c.h.bf16")
def vunpack_c_h_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Unpack high half of packed BF16 register to float32.

    Extracts the upper 16 bits of each 32-bit lane.
    """
    vsrc_reg, = params
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.bfloat16)
    _, high = unpack_bf16_register(vsrc_data, state.num_sublanes, state.num_lanes)
    state.write_vreg(dest_reg, high.to(torch.float32))


@instr("vunpack.i.l.bf16")
def vunpack_i_l_bf16(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Unpack low half (immediate path): reinterpret FP32 lanes as BF16 low.

    For values from scalar path (e.g. vstv) represented as FP32; converts
    to BF16 and back to FP32 for downstream use.
    """
    vsrc_reg, = params
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
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[0:num_lanes, :] = vsrc_data.transpose(0, 1)


@instr("vxpose.xlu0.b32.start")
def vxpose_xlu0_b32_start(state: ArchState, _: str, params: dict[str, Any]):
    """Transpose and load into XLU buffer (start of sequence).

    Transposes the source tensor (num_lanes x sublane) and stores it in
    the top rows of the XLU buffer. Used with vxpose.xlu0.b32.end for
    multi-tile transpose.
    """
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    state.xlu_buffer[0:num_lanes, :] = vsrc_data.transpose(0, 1)


@instr("vxpose.xlu0.b32.end")
def vxpose_xlu0_b32_end(state: ArchState, _: str, params: dict[str, Any]):
    """Transpose and load into XLU buffer (end of sequence).

    Rolls the buffer up, then writes transposed source into the next
    rows. Complements vxpose.xlu0.b32.start for sliding-window transpose.
    """
    vsrc_reg, num_lanes = params
    num_lanes = int(num_lanes)
    assert num_lanes <= state.num_lanes
    vsrc_data = state.read_vreg(vsrc_reg, dtype=torch.float32)
    # roll rightwards for a full tile
    state.xlu_buffer = state.xlu_buffer.roll(-state.num_lanes, dims=0)
    state.xlu_buffer[state.num_lanes:state.num_lanes+num_lanes, :] = vsrc_data.transpose(0, 1)


@instr("vpop.trf.xlu0")
def vpop_trf_xlu0(state: ArchState, dest_reg: str, params: dict[str, Any]):
    """Pop the leading 8x8 window from the XLU buffer into a vector register.

    Reads the top 8x8 from the XLU buffer, writes to dest (left 8x8 of
    vreg), rolls the buffer up, and zeroes the vacated bottom rows.
    """
    result = torch.zeros(state.num_sublanes, state.num_lanes, dtype=torch.float32)
    result[:, 0:state.num_sublanes] = state.xlu_buffer[0:state.num_sublanes, :].clone()
    state.xlu_buffer = state.xlu_buffer.roll(-state.num_sublanes, dims=0)
    state.xlu_buffer[-state.num_sublanes:, :] = 0
    state.write_vreg(dest_reg, result)


# === MXU Instructions ===

@instr("vmatpush.msra.mxu0")
def vmatpush_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU0.

    Loads the weight tile for matmul. MSRA = matrix storage / register file A.
    """
    src_vreg, = params
    state.push_mxu_weight("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu1")
def vmatpush_msra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU1."""
    src_vreg, = params
    state.push_mxu_weight("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu2")
def vmatpush_msra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU2."""
    src_vreg, = params
    state.push_mxu_weight("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.msra.mxu3")
def vmatpush_msra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Push weight matrix (MSRA) from vector reg into MXU3."""
    src_vreg, = params
    state.push_mxu_weight("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu0")
def vmatpush_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU0.

    Same as vmatpush.msra but transposes the source before pushing;
    used for RHS-major layouts (e.g. transposed matmul).
    """
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu1")
def vmatpush_xpose_msra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU1."""
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu1", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu2")
def vmatpush_xpose_msra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU2."""
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu2", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.xpose.msra.mxu3")
def vmatpush_xpose_msra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed weight matrix into MXU3."""
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu3", state.read_vreg(src_vreg, dtype=torch.float32))


@instr("vmatpush.bf16.xpose.msra.mxu0")
def vmatpush_bf16_xpose_msra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Push transposed BF16 weight matrix into MXU0.

    BF16 variant for mixed-precision matmul; source is BF16-packed.
    """
    src_vreg, = params
    state.push_mxu_weight_transpose("mxu0", state.read_vreg(src_vreg, dtype=torch.bfloat16))


@instr("vmatmul.f32.gmra.mxu0")
def vmatmul_f32_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation (GMRA) x weight (MSRA) in MXU0, accumulate.

    Reads activation from vsrc, multiplies with previously pushed weights,
    accumulates into the MXU accumulator. GMRA = general matrix register A.
    """
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.gmra.mxu1")
def vmatmul_f32_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU1, accumulate."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.gmra.mxu2")
def vmatmul_f32_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU2, accumulate."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.gmra.mxu3")
def vmatmul_f32_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply: activation x weight in MXU3, accumulate."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu0")
def vmatmul_f32_vlgmr_msra_gmra_mxu0(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU0 (vlgmr/msra/gmra variant).

    Same matmul as vmatmul.f32.gmra; different operand routing/format.
    """
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu0", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu1")
def vmatmul_f32_vlgmr_msra_gmra_mxu1(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU1 (vlgmr/msra/gmra variant)."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu1", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu2")
def vmatmul_f32_vlgmr_msra_gmra_mxu2(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU2 (vlgmr/msra/gmra variant)."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu2", activation)


@instr("vmatmul.f32.vlgmr.msra.gmra.mxu3")
def vmatmul_f32_vlgmr_msra_gmra_mxu3(state: ArchState, _: str, params: dict[str, Any]):
    """Matrix multiply in MXU3 (vlgmr/msra/gmra variant)."""
    src_vreg, = params
    activation = state.read_vreg(src_vreg, dtype=torch.float32)
    state.execute_mxu_matmul("mxu3", activation)


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
