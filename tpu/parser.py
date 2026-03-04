import re
from pathlib import Path
from dataclasses import dataclass


OperandValue = str | int | None


# One instruction: dest register, opcode, and argument fields (register part, immediate, or "")
@dataclass
class Instruction:
    opcode: str
    dest_reg: str
    args: dict[str, OperandValue]


# Memory allocation from original file: id, size in bytes, space, base address
@dataclass
class Allocation:
    id: str  # e.g. #allocation0
    size: int  # bytes (e.g. 0x2000)
    space: str  # vmem, sflag, hbm, etc.
    base_address: int = 0  # assigned by layout


class BundleParser:
    """Parse TPU compiler bundle dump files into instruction bundles."""

    # Matches instruction starts: "0xa   :  { ..." or "0xb LB: > { ..." or "0xc   : > { ..."
    _INSTRUCTION_START_RE = re.compile(
        r"^\s*(0x[0-9a-fA-F]+|\d+)\s*[^\{]*(\{.*)$"
    )
    # Matches: #allocation0 [shape = '...', space=vmem, size = 0x2000, ...] (stack3)
    _ALLOCATION_RE = re.compile(
        r"^\s*(#allocation\d+(?:_[A-Za-z0-9]+)?)\s+\[.*?space\s*=\s*(\w+).*?size\s*=\s*(0x[0-9a-fA-F]+|\d+).*?\]"
    )

    ALLOCATION_SUFFIX = "-post-delay-converter.txt"
    BUNDLE_SUFFIX = "-final_bundles.txt"

    def __init__(self) -> None:
        # EUP (elementwise unary pipeline) bookkeeping shared across bundles
        # so that producer/consumer pairs spanning different bundle addresses
        # can be lowered correctly.
        self._eup_value_producers: dict[str, tuple[str, str]] = {}
        self._ssa_token_to_vreg: dict[str, str] = {}

    _PREDICATED_TERNARY_SCALAR_OPS = {
        "sadd.s32",
        "ssub.s32",
        "sor.u32",
        "sand.u32",
        "scalar_lea.vmem",
        "scalar_lea.hbm",
        "scalar_lea.sflag",
    }
    _VECTOR_BINARY_OPS = {
        "vadd.f32",
        "vadd.s32",
        "vsub.s32",
        "vsub.f32",
        "vmul.f32",
        "vmul.u32",
        "vand.u32",
        "vor.u32",
        "vxor.u32",
        "vshll.u32",
        "vshrl.u32",
        "vcmp.lt.s32.totalorder",
        "vcmp.gt.s32.totalorder",
        "vcmp.eq.s32.totalorder",
        "vcmp.le.f32.partialorder",
        "vcmp.eq.f32.partialorder",
        "vc.u32",
        "vmor",
    }
    _VECTOR_UNARY_OPS = {
        "vmov",
        "vclz",
        "vcvt.s32.f32",
        "vrcp.f32",
        "vpow2.f32",
        "vpop.eup",
        "vweird.f32",
        "vtanh.f32",
        "vunpack.c.l.bf16",
        "vunpack.c.h.bf16",
        "vunpack.i.l.bf16",
        "vmatpush.msra.mxu0",
        "vmatpush.msra.mxu1",
        "vmatpush.msra.mxu2",
        "vmatpush.msra.mxu3",
        "vmatpush.xpose.msra.mxu0",
        "vmatpush.xpose.msra.mxu1",
        "vmatpush.xpose.msra.mxu2",
        "vmatpush.xpose.msra.mxu3",
        "vmatpush.bf16.xpose.msra.mxu0",
        "vmatmul.f32.gmra.mxu0",
        "vmatmul.f32.gmra.mxu1",
        "vmatmul.f32.gmra.mxu2",
        "vmatmul.f32.gmra.mxu3",
        "vmatmul.f32.vlgmr.msra.gmra.mxu0",
        "vmatmul.f32.vlgmr.msra.gmra.mxu1",
        "vmatmul.f32.vlgmr.msra.gmra.mxu2",
        "vmatmul.f32.vlgmr.msra.gmra.mxu3",
    }

    @classmethod
    def _operand_schema_for_opcode(cls, opcode: str) -> list[str] | None:
        if opcode.startswith("inlined_call_operand."):
            return ["imm1"]
        if opcode.startswith("int_to_ptr."):
            return ["rs1"]
        if opcode in ("vsyncpa", "vsyncadd"):
            return ["pred", "addr", "rs1"]
        if opcode in ("dma.hbm_to_vmem", "dma.vmem_to_hbm"):
            return ["pred", "rs1", "imm1", "rs2", "sync"]
        if opcode == "dma.done.wait":
            return ["pred", "sync"]
        if opcode == "smov":
            return ["pred", "rs1", "rs2"]
        if opcode in ("sshll.u32", "sshra.s32"):
            return ["pred", "rs1", "imm1"]
        if opcode in cls._PREDICATED_TERNARY_SCALAR_OPS:
            return ["pred", "rs1", "rs2"]
        if opcode == "scalar_select":
            return ["pred", "rs1", "rs2"]
        if opcode == "sphi":
            return ["rs1"]
        if opcode in (
            "scmp.eq.s32.totalorder",
            "scmp.ne.s32.totalorder",
            "scmp.ge.s32.totalorder",
            "scmp.lt.s32.totalorder",
        ):
            return ["rs1", "rs2"]
        if opcode in ("por", "pnand"):
            return ["ps1", "ps2"]
        if opcode == "pneg":
            return ["ps1"]
        if opcode == "sst":
            return ["addr", "rs1"]
        if opcode == "sld":
            return ["addr"]
        if opcode == "sbr.rel":
            return ["pred", "target"]
        if opcode == "shalt.err":
            return ["pred"]
        if opcode == "vstv":
            return ["rs1"]
        if opcode == "vld":
            return ["addr", "sm", "ss"]
        if opcode in ("vst", "vst.msk"):
            return ["addr", "sm", "vs1"]
        if opcode in cls._VECTOR_BINARY_OPS:
            return ["vs1", "vs2"]
        if opcode in cls._VECTOR_UNARY_OPS:
            return ["vs1"]
        if opcode == "vsel":
            return ["vm1", "vs1", "vs2"]
        if opcode == "vlaneseq":
            return []
        if opcode == "vset.pattern.permute.xlu0":
            return ["imm1"]
        if opcode == "vperm.xlu0":
            return ["vs1", "imm1"]
        if opcode == "vpop.permute.xlu0":
            return ["imm1"]
        if opcode == "vrot.slane":
            return ["vs1", "imm1"]
        if opcode == "vcmask":
            return ["imm1", "imm2"]
        if opcode == "vpack.c.bf16":
            return ["vs1", "vs2"]
        if opcode in (
            "vxpose.xlu0.b32.start.end",
            "vxpose.xlu0.b32.start",
            "vxpose.xlu0.b32.end",
        ):
            return ["vs1", "imm1"]
        if opcode == "vpop.trf.xlu0":
            return ["imm1"]
        if opcode == "vmatmul.msk.f32.vlgmr.msra.gmra.mxu0":
            return ["vm1", "vs1"]
        if opcode in (
            "vpop.f32.mrf.mxu0",
            "vpop.f32.mrf.mxu1",
            "vpop.f32.mrf.mxu2",
            "vpop.f32.mrf.mxu3",
        ):
            return []
        return None

    @staticmethod
    def _default_operand_value(field: str) -> OperandValue:
        return None if field == "pred" or field.startswith("ps") else 0

    @staticmethod
    def _infer_operand_field(token: str, counts: dict[str, int]) -> str:
        if token.startswith("!p") or token.startswith("p"):
            key = "pred"
        elif token.startswith("vm"):
            counts["vm"] = counts.get("vm", 0) + 1
            key = f"vm{counts['vm']}"
        elif token.startswith("v"):
            counts["vs"] = counts.get("vs", 0) + 1
            key = f"vs{counts['vs']}"
        elif token.startswith("ss="):
            counts["ss"] = counts.get("ss", 0) + 1
            key = f"ss{counts['ss']}"
        elif token.startswith("s"):
            counts["rs"] = counts.get("rs", 0) + 1
            key = f"rs{counts['rs']}"
        else:
            counts["imm"] = counts.get("imm", 0) + 1
            key = f"imm{counts['imm']}"
        return key

    @classmethod
    def _build_operand_dict(cls, opcode: str, args: list[str]) -> dict[str, OperandValue]:
        schema = cls._operand_schema_for_opcode(opcode)
        if schema is not None:
            values: dict[str, OperandValue] = {
                field: cls._default_operand_value(field) for field in schema
            }
            active_schema = schema
            if schema and schema[0] == "pred" and args:
                first = args[0]
                if not (first.startswith("p") or first.startswith("!p")):
                    active_schema = schema[1:]
            for i, value in enumerate(args):
                if i < len(active_schema):
                    field = active_schema[i]
                else:
                    field = f"imm{i + 1}"
                    values.setdefault(field, cls._default_operand_value(field))
                values[field] = value
            return values

        inferred: dict[str, OperandValue] = {}
        counts: dict[str, int] = {}
        for value in args:
            field = cls._infer_operand_field(value, counts)
            while field in inferred:
                counts["imm"] = counts.get("imm", 0) + 1
                field = f"imm{counts['imm']}"
            inferred[field] = value
        return inferred

    # --- File discovery ---

    @staticmethod
    def _find_file_by_suffix(
        partial_path: Path, suffix: str, prefer_exact: bool = True
    ) -> Path:
        """Resolve partial path to a file matching {stem}-*-{suffix}."""
        partial_path = Path(partial_path)
        if partial_path.is_file():
            return partial_path
        parent = partial_path.parent
        stem = partial_path.name
        candidates = list(parent.glob(f"{stem}-*{suffix}"))
        if prefer_exact:
            exact = [
                p
                for p in candidates
                if re.fullmatch(rf"{re.escape(stem)}-\d+{re.escape(suffix)}", p.name)
            ]
            chosen = exact[0] if exact else (candidates[0] if candidates else None)
        else:
            chosen = candidates[0] if candidates else None
        if chosen is None:
            raise FileNotFoundError(f"No *{suffix} file found for {partial_path}")
        return chosen

    # --- Symbol table (allocations) ---

    def _parse_symbol_table(self, partial_path: Path) -> dict[str, Allocation]:
        """Parse *-original.txt and return allocation id -> Allocation (with base_address)."""
        file_path = self._find_file_by_suffix(partial_path, self.ALLOCATION_SUFFIX)
        allocations: list[Allocation] = []

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                m = self._ALLOCATION_RE.search(line)
                if not m:
                    continue
                alloc_id, space, size_str = m.group(1), m.group(2), m.group(3)
                size = int(size_str, 16) if size_str.startswith("0x") else int(size_str)
                allocations.append(
                    Allocation(id=alloc_id, size=size, space=space, base_address=0)
                )

        # Contiguous layout per space
        offset_by_space: dict[str, int] = {}
        for alloc in allocations:
            base = offset_by_space.get(alloc.space, 0)
            alloc.base_address = base
            offset_by_space[alloc.space] = base + alloc.size

        return {a.id: a for a in allocations}

    def parse_allocations(self, partial_path: Path) -> dict[str, Allocation]:
        """Parse *-original.txt and return allocation id -> Allocation (public API)."""
        return self._parse_symbol_table(partial_path)

    # --- Instruction parsing ---

    @staticmethod
    def _register_part(var: str) -> str:
        """Extract register name.

        For coalesced names (e.g. %s114_s0), return suffix after the final
        underscore ("s0"). For temporary SSA ids without underscores (e.g.
        %15), keep the full token so producer/consumer links are preserved.
        """
        if "_" in var:
            return var.rsplit("_", 1)[1]
        return var

    @staticmethod
    def _split_args_top_level(s: str) -> list[str]:
        """Split argument string by comma, respecting brackets [] and ()."""
        out: list[str] = []
        depth = 0
        start = 0
        for i, c in enumerate(s):
            if c in "[(":
                depth += 1
            elif c in "])":
                depth -= 1
            elif c == "," and depth == 0:
                out.append(s[start:i].strip())
                start = i + 1
        out.append(s[start:].strip())
        return out

    # vmem: [#allocation0] or [#allocation0 + $0x8]
    _VMEM_ALLOC_RE = re.compile(
        r"vmem:\s*\[(#allocation\d+(?:_[A-Za-z0-9]+)?)(?:\s*\+\s*\$?(0x[0-9a-fA-F]+|\d+))?\]"
    )
    # vmem: [%s165_s1] or [%s165_s1 + $0x78] (register-based address)
    _VMEM_REG_RE = re.compile(
        r"vmem:\s*\[%([\w]+)(?:\s*\+\s*\$?(0x[0-9a-fA-F]+|\d+))?\]"
    )
    # smem: [#allocation8_spill] or [#allocation0 + $0x4]
    _SMEM_ALLOC_RE = re.compile(
        r"smem:\s*\[(#allocation\d+(?:_[A-Za-z0-9]+)?)(?:\s*\+\s*\$?(0x[0-9a-fA-F]+|\d+))?\]"
    )
    # sm: mask immediate for vld/vst, e.g. sm:$0xf or sm:$0xff
    _VMEM_SMASK_IMM_RE = re.compile(r"sm:\s*\$?(0x[0-9a-fA-F]+|\d+)\b")
    # ss: sublane stride immediate, e.g. ss:$0
    _VMEM_SSTRIDE_IMM_RE = re.compile(r"ss:\s*\$?(0x[0-9a-fA-F]+|\d+)\b")

    # inlined_call_operand: [shape: f32[8,128], index: 0, kind: input, ...]
    _INLINED_CALL_OPERAND_RE = re.compile(
        r"\[shape:\s*(\w+)\[([^\]]*)\],\s*index:\s*(\d+)"
    )

    _DTYPE_BYTES: dict[str, int] = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "s32": 4,
        "u32": 4,
        "s16": 2,
        "u16": 2,
        "s8": 1,
        "u8": 1,
    }

    @classmethod
    def _inlined_call_operand_info(
        cls, rest: str
    ) -> tuple[str, list[int], int, int] | None:
        """Parse inlined_call_operand metadata.

        Returns (dtype, dims, index, logical_size_bytes), or None if not parseable.
        """
        m = cls._INLINED_CALL_OPERAND_RE.search(rest)
        if not m:
            return None
        dtype_str, dims_str, index_str = m.group(1), m.group(2), m.group(3)
        index = int(index_str)
        elem_bytes = cls._DTYPE_BYTES.get(dtype_str, 4)
        if not dims_str.strip():
            elem_count = 1
            dims: list[int] = []
        else:
            dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
            elem_count = 1
            for d in dims:
                elem_count *= d
        operand_size = elem_count * elem_bytes
        return (dtype_str, dims, index, operand_size)

    @classmethod
    def _inlined_call_operand_base_address(cls, rest: str) -> int | None:
        """Compute HBM base address for inlined_call_operand. Returns byte address or None."""
        info = cls._inlined_call_operand_info(rest)
        if info is None:
            return None
        _dtype, _dims, index, size = info
        return index * size

    @staticmethod
    def _physical_vmem_operand_size(dtype_str: str, dims: list[int], logical_size: int) -> int:
        """Return physical VMEM bytes for an inlined operand.

        TPU BF16 RHS tiles used by MXU matmul are consumed through coalesced
        load paths as full register images (4096 B), even for logical [K, 8]
        tensors whose element payload is 2048 B.
        """
        if dtype_str == "bf16" and len(dims) == 2 and dims[1] == 8:
            return max(logical_size, 4096)
        return logical_size

    def _parse_arg_values(self, seg: str) -> list[str]:
        """Extract all arg values from a segment; returns list for vld/vst compound args."""
        seg = re.sub(r"^/\*[^*]*\*/\s*", "", seg.strip())
        values: list[str] = []

        pred_src_match = re.match(r"^\(\s*(!?)%([\w]+)\s*,\s*%([\w]+)\s*\)$", seg)
        if pred_src_match:
            pred_sign, pred_var, src_var = pred_src_match.groups()
            pred_reg = self._register_part(pred_var)
            src_reg = self._register_part(src_var)
            values.append(f"!{pred_reg}" if pred_sign else pred_reg)
            values.append(src_reg)
            return values

        pred_only_match = re.match(r"^\(\s*(!?)%([\w]+)\s*\)$", seg)
        if pred_only_match:
            pred_sign, pred_var = pred_only_match.groups()
            pred_reg = self._register_part(pred_var)
            return [f"!{pred_reg}" if pred_sign else pred_reg]

        # vld/vst: [vmem:[#allocation0] ...] or [vmem:[#allocation0 + $0x8] ...]
        vmem_match = self._VMEM_ALLOC_RE.search(seg)
        if vmem_match:
            alloc_id = vmem_match.group(1)
            offset = vmem_match.group(2)
            if offset:
                offset_int = int(offset, 16) if offset.startswith("0x") else int(offset)
                values.append(f"{alloc_id}+{offset_int}")
            else:
                values.append(alloc_id)
            # sm: mask immediate for vld/vst, e.g. sm:$0xf
            smask_match = self._VMEM_SMASK_IMM_RE.search(seg)
            if smask_match:
                imm_str = smask_match.group(1)
                imm_val = int(imm_str, 16) if imm_str.startswith("0x") else int(imm_str)
                values.append(str(imm_val))
            sstride_match = self._VMEM_SSTRIDE_IMM_RE.search(seg)
            if sstride_match:
                imm_str = sstride_match.group(1)
                imm_val = int(imm_str, 16) if imm_str.startswith("0x") else int(imm_str)
                values.append(f"ss={imm_val}")

        # vld/vst: vmem:[%sX_sY] or vmem:[%s165_s1 + $0x78] (register-based address)
        if not values and "vmem:" in seg:
            reg_match = self._VMEM_REG_RE.search(seg)
            if reg_match:
                var = reg_match.group(1)
                reg = var.rsplit("_", 1)[1] if "_" in var else ""
                offset = reg_match.group(2)
                if reg:
                    if offset:
                        offset_int = int(offset, 16) if offset.startswith("0x") else int(offset)
                        values.append(f"{reg}+{offset_int}")
                    else:
                        values.append(reg)
                smask_match = self._VMEM_SMASK_IMM_RE.search(seg)
                if smask_match:
                    imm_str = smask_match.group(1)
                    imm_val = int(imm_str, 16) if imm_str.startswith("0x") else int(imm_str)
                    values.append(str(imm_val))
                sstride_match = self._VMEM_SSTRIDE_IMM_RE.search(seg)
                if sstride_match:
                    imm_str = sstride_match.group(1)
                    imm_val = int(imm_str, 16) if imm_str.startswith("0x") else int(imm_str)
                    values.append(f"ss={imm_val}")

        # sld/sst: [smem:[#allocationN(_spill)] ...]
        if not values and "smem:" in seg:
            smem_match = self._SMEM_ALLOC_RE.search(seg)
            if smem_match:
                alloc_id = smem_match.group(1)
                offset = smem_match.group(2)
                if offset:
                    offset_int = int(offset, 16) if offset.startswith("0x") else int(offset)
                    values.append(f"{alloc_id}+{offset_int}")
                else:
                    values.append(alloc_id)
                reg_matches = re.findall(r"%([\w]+)", seg)
                if reg_matches:
                    reg = self._register_part(reg_matches[-1])
                    if reg:
                        values.append(reg)

        # vst: /*vst_source=*/%v18_v2 -> source register
        if "vst_source" in seg or "/*vst_source=*/" in seg:
            src_match = re.search(r"vst_source=\*/\s*%([\w]+)", seg)
            if src_match is None:
                all_vars = re.findall(r"%([\w]+)", seg)
                var = all_vars[-1] if all_vars else ""
            else:
                var = src_match.group(1)
            reg = var.rsplit("_", 1)[1] if "_" in var else ""
            if reg:
                values.append(reg)

        if values:
            return values

        # Non-vmem segment: single value
        single = self._parse_arg_value_simple(seg)
        return [single] if single else []

    @staticmethod
    def _parse_arg_value_simple(seg: str) -> str:
        """Extract single arg value for non-vld/vst segments."""
        var_match = re.search(r"(!?)%([\w]+)", seg)
        if var_match:
            sign, var = var_match.group(1), var_match.group(2)
            reg = var.rsplit("_", 1)[1] if "_" in var else var
            if sign and reg:
                return f"!{reg}"
            return reg
        if re.match(
            r"^[+-]?(?:nan|inf|(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)$",
            seg,
            re.IGNORECASE,
        ):
            return seg
        if re.match(r"^-?(?:0x[0-9a-fA-F]+|\d+)$", seg):
            return seg
        alloc_match = re.search(r"\[(#allocation\d+(?:_[A-Za-z0-9]+)?)\]", seg)
        if alloc_match:
            return alloc_match.group(1)
        # [#allocationN + $0xM] or [#allocationN + $M]
        alloc_offset_match = re.search(
            r"\[(#allocation\d+(?:_[A-Za-z0-9]+)?)\s*\+\s*\$?(?:0x([0-9a-fA-F]+)|(\d+))\]",
            seg,
        )
        if alloc_offset_match:
            alloc_id = alloc_offset_match.group(1)
            hex_off, dec_off = alloc_offset_match.group(2), alloc_offset_match.group(3)
            offset_int = int(hex_off, 16) if hex_off else int(dec_off)
            return f"{alloc_id}+{offset_int}"
        if re.match(r"^\$0x[0-9a-fA-F]+$", seg) or seg in ("$0",):
            return seg
        return ""

    @staticmethod
    def _resolve_args(
        args: dict[str, OperandValue], symbol_table: dict[str, Allocation]
    ) -> dict[str, OperandValue]:
        """Replace #allocation refs (and #alloc+offset) with resolved addresses."""
        result: dict[str, OperandValue] = {}

        def normalize_alloc_id(alloc_id: str) -> str:
            if alloc_id in symbol_table:
                return alloc_id
            m = re.match(r"^(#allocation\d+)(?:_[A-Za-z0-9]+)?$", alloc_id)
            if m and m.group(1) in symbol_table:
                return m.group(1)
            return alloc_id

        for key, value in args.items():
            if not isinstance(value, str):
                result[key] = value
                continue
            a = value
            alloc_id = normalize_alloc_id(a)
            if alloc_id in symbol_table:
                result[key] = str(symbol_table[alloc_id].base_address)
            elif re.match(r"^#allocation\d+(?:_[A-Za-z0-9]+)?\+\d+$", a):
                alloc_id, offset_str = a.rsplit("+", 1)
                alloc_id = normalize_alloc_id(alloc_id)
                base = symbol_table[alloc_id].base_address if alloc_id in symbol_table else 0
                result[key] = str(base + int(offset_str))
            else:
                result[key] = a
        return result

    def _parse_one_instruction(self, instr_str: str) -> Instruction | None:
        """Parse a single instruction '%dest = opcode args...' into dest_reg, opcode, args."""
        instr_str = instr_str.strip().lstrip("> {").strip().removesuffix(" }").strip()
        instr_str = instr_str.rstrip()
        if not instr_str:
            return None
        if instr_str.endswith("*/"):
            last_close = instr_str.rfind("*/")
            last_open = instr_str.rfind("/*", 0, last_close)
            if last_open != -1 and "*/" not in instr_str[last_open : last_close]:
                instr_str = instr_str[:last_open].rstrip()
        # Opcode can be \S+ or "inlined_call_operand.<no memory space>" (contains spaces)
        m = re.match(
            r"^\s*%(\S+)\s*=\s*(inlined_call_operand\.<no\s+memory\s+space>|inlined_call_operand\.(?:hbm|vmem)|\S+)\s*(.*)$",
            instr_str,
        )
        if not m:
            print(f"Failed to parse instruction: {instr_str}")
            return None
        dest_var, opcode, rest = m.group(1), m.group(2), m.group(3)
        dest_reg = self._register_part(dest_var)

        # inlined_call_operand: use #operand{index} so _resolve_args gets cumulative address
        if opcode.startswith("inlined_call_operand."):
            info = self._inlined_call_operand_info(rest)
            if info is not None:
                _dtype, _dims, index, _size = info
                return Instruction(
                    dest_reg=dest_reg,
                    opcode=opcode,
                    args=self._build_operand_dict(opcode, [f"#operand{index}"]),
                )
            # fall through to generic parsing if shape/index not found

        if opcode == "sbr.rel":
            args: list[str] = []
            pred_match = re.search(r"(!?)%([\w]+)", rest)
            if pred_match:
                sign, pred_var = pred_match.groups()
                pred_reg = self._register_part(pred_var)
                args.append(f"!{pred_reg}" if sign else pred_reg)
            target_match = re.search(r"target bundleno\s*=\s*(0x[0-9a-fA-F]+|\d+)", rest)
            if target_match:
                args.append(target_match.group(1))
            return Instruction(dest_reg=dest_reg, opcode=opcode, args=self._build_operand_dict(opcode, args))

        arg_segments = self._split_args_top_level(rest)
        args: list[str] = []
        for seg in arg_segments:
            args.extend(self._parse_arg_values(seg))
        return Instruction(dest_reg=dest_reg, opcode=opcode, args=self._build_operand_dict(opcode, args))

    # --- Bundle parsing ---

    def _parse_bundle_raw(self, payload: str) -> list[Instruction]:
        """Parse bundle payload into instructions (no symbol resolution).

        This pass also performs light SSA->physical lowering for the EUP
        (elementwise unary pipeline) instructions that the TPU compiler
        now emits in two-stage form:

          %t = vrcp.f32 %vX
          %vy = vpop.eup %t

        or

          %t = vtanh.f32 %vX
          %vy = vpop.eup %t

        or

          %t = vpow2.f32 %vX
          %vy = vpop.eup %t

        In the simulator we model these as single vector ops:

          vrcp.f32 vy, [vX]
          vtanh.f32 vy, [vX]

        and for vpow2.f32 we follow the hardware behaviour and overwrite
        the source register in-place, with vpop.eup acting as a plain
        move from that source into its destination:

          vpow2.f32 vX, [vX]
          vpop.eup  vy, [vX]
        """
        payload = re.sub(r"\s*\}\s*/\*.*\*/\s*$", "", payload.strip()).strip()
        if payload.endswith(" }"):
            payload = payload[:-2].strip()
        elif payload.endswith("}"):
            payload = payload[:-1].strip()
        result: list[Instruction] = []

        # First pass: parse each instruction and collect EUP metadata.
        raw_instrs: list[tuple[str, Instruction]] = []
        for part in (p.strip() for p in payload.split(";;") if p.strip()):
            instr = self._parse_one_instruction(part)
            if instr is None:
                continue

            # Extract SSA destination (if any) for EUP plumbing.
            header = part.strip().lstrip("> {").strip().removesuffix(" }").strip()
            m = re.match(r"^\s*%(\S+)\s*=", header)
            ssa_dest = m.group(1) if m else None

            opcode = instr.opcode

            # Instructions whose SSA destination is not a real register (e.g. stores).
            # For these, the LLO dest is just a token; the simulator ISA handler does
            # not write back to a register, so we clear dest_reg to avoid inventing
            # bogus registers like "62" for vector stores.
            if opcode in ("vst", "vst.msk"):
                instr.dest_reg = ""

            # EUP value producers: vrcp / vtanh – effect realized at vpop.eup.
            if ssa_dest and opcode in ("vrcp.f32", "vtanh.f32"):
                src_reg = instr.args.get("vs1", 0)
                if isinstance(src_reg, str) and src_reg:
                    self._eup_value_producers[ssa_dest] = (opcode, src_reg)
                # Do NOT emit this instruction now; it will be materialized
                # as a concrete vector op when we see the matching vpop.eup.
                continue

            # vpow2.f32: overwrite source vector register in-place. The SSA
            # destination is an EUP token, not a hardware register.
            if ssa_dest and opcode == "vpow2.f32":
                src_reg = instr.args.get("vs1", 0)
                if isinstance(src_reg, str) and src_reg:
                    self._ssa_token_to_vreg[ssa_dest] = src_reg
                    instr.dest_reg = src_reg

            raw_instrs.append((ssa_dest or "", instr))

        # Second pass: lower vpop.eup and emit final instruction list.
        for ssa_dest, instr in raw_instrs:
            if instr.opcode == "vpop.eup":
                token = instr.args.get("vs1", 0)
                if not isinstance(token, str) or not token:
                    result.append(instr)
                    continue

                # Case 1: token refers to a vrcp/vtanh EUP producer.
                if token in self._eup_value_producers:
                    prod_opcode, src_reg = self._eup_value_producers.pop(token)
                    result.append(
                        Instruction(
                            opcode=prod_opcode,
                            dest_reg=instr.dest_reg,
                            args=self._build_operand_dict(prod_opcode, [src_reg]),
                        )
                    )
                    continue

                # Case 2: token refers to a vpow2.f32 in-place producer.
                if token in self._ssa_token_to_vreg:
                    src_reg = self._ssa_token_to_vreg[token]
                    result.append(
                        Instruction(
                            opcode="vpop.eup",
                            dest_reg=instr.dest_reg,
                            args=self._build_operand_dict("vpop.eup", [src_reg]),
                        )
                    )
                    continue

                # Fallback: treat argument as already-physical.
                result.append(instr)
            else:
                result.append(instr)
        return result

    def parse_bundle(
        self, payload: str, symbol_table: dict[str, Allocation] | None = None
    ) -> list[Instruction]:
        """Parse bundle payload and return instructions with named operand dictionaries."""
        instructions = self._parse_bundle_raw(payload)
        if symbol_table:
            for instr in instructions:
                instr.args = self._resolve_args(instr.args, symbol_table)
        return instructions

    # --- Program parsing ---

    def _iter_bundle_payloads(self, partial_path: Path):
        """Yield (address, payload) for each bundle in the file."""
        current_address: str | None = None
        current_lines: list[str] = []

        file_path = self._find_file_by_suffix(partial_path, self.BUNDLE_SUFFIX)

        with file_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.rstrip("\n")

                if current_address is not None:
                    current_lines.append(line.strip())
                    if "}" in line:
                        payload = " ".join(current_lines).strip()
                        yield int(current_address, 16), payload
                        current_address = None
                        current_lines = []
                    continue

                m = self._INSTRUCTION_START_RE.match(line)
                if m is None:
                    continue
                address, payload_start = m.group(1), m.group(2).strip()

                if "}" in payload_start:
                    yield int(address, 16), payload_start
                    continue

                current_address = address
                current_lines = [payload_start]

    # Match inlined_call_operand.hbm, .vmem, or .<no memory space>
    _INLINED_CALL_OPERAND_INSTR_RE = re.compile(
        r"^\s*%\S+\s*=\s*inlined_call_operand\.(?:hbm|vmem|<no\s+memory\s+space>)\s*(.*)$"
    )

    def _collect_operands_into_symbol_table(
        self, payload: str, symbol_table: dict[str, Allocation]
    ) -> None:
        """Scan bundle payload for inlined_call_operand and add #operand{N} to symbol_table.

        HBM and VMEM operands are stacked in their respective spaces.
        <no memory space> (scalars) are not stacked.
        """
        payload = re.sub(r"\s*\}\s*/\*.*\*/\s*$", "", payload.strip()).strip()
        payload = re.sub(r"^\s*\{\s*", "", payload)
        if payload.endswith(" }"):
            payload = payload[:-2].strip()
        elif payload.endswith("}"):
            payload = payload[:-1].strip()

        # Collect (index, size, memory_space) for each operand
        # memory_space: 'hbm' | 'vmem' | 'smem' (scalar)
        operands: list[tuple[int, int, str]] = []
        for part in (p.strip() for p in payload.split(";;") if p.strip()):
            m = self._INLINED_CALL_OPERAND_INSTR_RE.match(part)
            if not m:
                continue
            rest = m.group(1)
            info = self._inlined_call_operand_info(rest)
            if info is None:
                continue
            dtype_str, dims, index, size = info
            if "inlined_call_operand.hbm" in part:
                space = "hbm"
            elif "inlined_call_operand.vmem" in part:
                space = "vmem"
                size = self._physical_vmem_operand_size(dtype_str, dims, size)
            else:
                space = "smem"  # <no memory space>
            operands.append((index, size, space))

        # Assign base addresses: HBM and VMEM stacked separately; smem (scalars) excluded.
        # Start after already-materialized allocations in each space so
        # inlined operands do not alias temporary buffers.
        operands.sort(key=lambda x: x[0])
        hbm_offset = max(
            (alloc.base_address + alloc.size for alloc in symbol_table.values() if alloc.space == "hbm"),
            default=0,
        )
        vmem_offset = max(
            (alloc.base_address + alloc.size for alloc in symbol_table.values() if alloc.space == "vmem"),
            default=0,
        )
        for index, size, space in operands:
            op_id = f"#operand{index}"
            if op_id not in symbol_table:
                if space == "hbm":
                    base_address = hbm_offset
                    hbm_offset += size
                elif space == "vmem":
                    base_address = vmem_offset
                    vmem_offset += size
                else:
                    base_address = 0  # scalar, not stacked
                symbol_table[op_id] = Allocation(
                    id=op_id, size=size, space=space, base_address=base_address
                )

    def parse_program(
        self, partial_path: Path
    ) -> tuple[dict[str, Allocation], dict[int, list[Instruction]]]:
        """Parse program from partial path; resolves #allocation refs to base addresses."""
        # Reset EUP bookkeeping for each new program.
        self._eup_value_producers.clear()
        self._ssa_token_to_vreg.clear()
        symbol_table = self._parse_symbol_table(partial_path)

        bundles: dict[int, list[Instruction]] = {}
        for addr, payload in self._iter_bundle_payloads(partial_path):
            self._collect_operands_into_symbol_table(payload, symbol_table)
            bundles[addr] = self.parse_bundle(payload, symbol_table)

        return symbol_table, bundles
