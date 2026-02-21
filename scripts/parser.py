import re
from pathlib import Path
from dataclasses import dataclass


# One instruction: dest register, opcode, and argument fields (register part, immediate, or "")
@dataclass
class Instruction:
    opcode: str
    dest_reg: str
    args: list[str]


# Memory allocation from original file: id, size in bytes, space, base address
@dataclass
class Allocation:
    id: str  # e.g. #allocation0
    size: int  # bytes (e.g. 0x2000)
    space: str  # vmem, sflag, hbm, etc.
    base_address: int = 0  # assigned by layout


class BundleParser:
    """Parse TPU compiler bundle dump files into instruction bundles."""

    # Matches instruction starts like "0xa   :  { %94 = ..." or multiline "0x3   :  { %12 = ..."
    _INSTRUCTION_START_RE = re.compile(r"^\s*(0x[0-9a-fA-F]+|\d+)\s*:\s*(\{.*)$")
    # Matches: #allocation0 [shape = '...', space=vmem, size = 0x2000, ...] (stack3)
    _ALLOCATION_RE = re.compile(
        r"^\s*(#allocation\d+)\s+\[.*?space\s*=\s*(\w+).*?size\s*=\s*(0x[0-9a-fA-F]+|\d+).*?\]"
    )

    ALLOCATION_SUFFIX = "-original.txt"
    BUNDLE_SUFFIX = "-final_bundles.txt"

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
        """Extract register name: part after last underscore (e.g. %s114_s0 -> s0), else ""."""
        if "_" in var:
            return var.rsplit("_", 1)[1]
        return ""

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
        r"vmem:\s*\[(#allocation\d+)(?:\s*\+\s*\$?(0x[0-9a-fA-F]+|\d+))?\]"
    )

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
    def _inlined_call_operand_info(cls, rest: str) -> tuple[int, int, int] | None:
        """Parse inlined_call_operand [shape: ..., index: ...] -> (base_address, size, index).

        Operands are stacked contiguously: operand 0 at 0x0, operand 1 at
        sizeof(operand 0), etc. Returns (base_address_bytes, operand_size_bytes, index)
        or None if not parseable.
        """
        m = cls._INLINED_CALL_OPERAND_RE.search(rest)
        if not m:
            return None
        dtype_str, dims_str, index_str = m.group(1), m.group(2), m.group(3)
        index = int(index_str)
        elem_bytes = cls._DTYPE_BYTES.get(dtype_str, 4)
        if not dims_str.strip():
            elem_count = 1
        else:
            dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
            elem_count = 1
            for d in dims:
                elem_count *= d
        operand_size = elem_count * elem_bytes
        base_address = index * operand_size
        return (base_address, operand_size, index)

    @classmethod
    def _inlined_call_operand_base_address(cls, rest: str) -> int | None:
        """Compute HBM base address for inlined_call_operand. Returns byte address or None."""
        info = cls._inlined_call_operand_info(rest)
        return info[0] if info else None

    def _parse_arg_values(self, seg: str) -> list[str]:
        """Extract all arg values from a segment; returns list for vld/vst compound args."""
        seg = re.sub(r"^/\*[^*]*\*/\s*", "", seg.strip())
        values: list[str] = []

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

        # vst: /*vst_source=*/%v18_v2 -> source register
        if "vst_source" in seg or "/*vst_source=*/" in seg:
            var_match = re.search(r"%([\w]+)", seg)
            if var_match:
                var = var_match.group(1)
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
        var_match = re.search(r"%([\w]+)", seg)
        if var_match:
            var = var_match.group(1)
            return var.rsplit("_", 1)[1] if "_" in var else ""
        if re.match(r"^-?\d+\.\d+$", seg):
            return seg
        if re.match(r"^-?(?:0x[0-9a-fA-F]+|\d+)$", seg):
            return seg
        alloc_match = re.search(r"\[(#allocation\d+)\]", seg)
        if alloc_match:
            return alloc_match.group(1)
        if re.match(r"^\$0x[0-9a-fA-F]+$", seg) or seg in ("$0",):
            return seg
        return ""

    @staticmethod
    def _resolve_args(
        args: list[str], symbol_table: dict[str, Allocation]
    ) -> list[str]:
        """Replace #allocation refs (and #alloc+offset) with resolved addresses."""
        result: list[str] = []
        for a in args:
            if a in symbol_table:
                result.append(str(symbol_table[a].base_address))
            elif re.match(r"^#allocation\d+\+\d+$", a):
                alloc_id, offset_str = a.rsplit("+", 1)
                base = symbol_table[alloc_id].base_address if alloc_id in symbol_table else 0
                result.append(str(base + int(offset_str)))
            else:
                result.append(a)
        return result

    def _parse_one_instruction(self, instr_str: str) -> Instruction | None:
        """Parse a single instruction '%dest = opcode args...' into dest_reg, opcode, args."""
        instr_str = instr_str.strip().lstrip("{").strip().removesuffix(" }").strip()
        instr_str = instr_str.rstrip()
        if instr_str.endswith("*/"):
            last_close = instr_str.rfind("*/")
            last_open = instr_str.rfind("/*", 0, last_close)
            if last_open != -1 and "*/" not in instr_str[last_open : last_close]:
                instr_str = instr_str[:last_open].rstrip()
        # Opcode can be \S+ or "inlined_call_operand.<no memory space>" (contains spaces)
        m = re.match(
            r"^\s*%(\S+)\s*=\s*(inlined_call_operand\.<no\s+memory\s+space>|inlined_call_operand\.hbm|\S+)\s*(.*)$",
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
                _base, _size, index = info
                return Instruction(
                    dest_reg=dest_reg, opcode=opcode, args=[f"#operand{index}"]
                )
            # fall through to generic parsing if shape/index not found

        arg_segments = self._split_args_top_level(rest)
        args: list[str] = []
        for seg in arg_segments:
            args.extend(self._parse_arg_values(seg))
        return Instruction(dest_reg=dest_reg, opcode=opcode, args=args)

    # --- Bundle parsing ---

    def _parse_bundle_raw(self, payload: str) -> list[Instruction]:
        """Parse bundle payload into instructions (no symbol resolution)."""
        payload = re.sub(r"\s*\}\s*/\*.*\*/\s*$", "", payload.strip()).strip()
        if payload.endswith(" }"):
            payload = payload[:-2].strip()
        elif payload.endswith("}"):
            payload = payload[:-1].strip()
        result: list[Instruction] = []
        for part in (p.strip() for p in payload.split(";;") if p.strip()):
            parsed = self._parse_one_instruction(part)
            if parsed is not None:
                result.append(parsed)
        return result

    def parse_bundle(
        self, payload: str, symbol_table: dict[str, Allocation] | None = None
    ) -> list[Instruction]:
        """Parse bundle payload; resolve #allocation refs if symbol_table given."""
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

    # Match inlined_call_operand.hbm or inlined_call_operand.<no memory space>
    _INLINED_CALL_OPERAND_INSTR_RE = re.compile(
        r"^\s*%\S+\s*=\s*inlined_call_operand\.(?:hbm|<no\s+memory\s+space>)\s*(.*)$"
    )

    def _collect_operands_into_symbol_table(
        self, payload: str, symbol_table: dict[str, Allocation]
    ) -> None:
        """Scan bundle payload for inlined_call_operand and add #operand{N} to symbol_table.

        Base addresses are cumulative: operand 0 at 0, operand 1 at sizeof(0), etc.
        Only HBM operands get space='hbm'; <no memory space> (scalars) get space='smem'.
        """
        payload = re.sub(r"\s*\}\s*/\*.*\*/\s*$", "", payload.strip()).strip()
        payload = re.sub(r"^\s*\{\s*", "", payload)
        if payload.endswith(" }"):
            payload = payload[:-2].strip()
        elif payload.endswith("}"):
            payload = payload[:-1].strip()

        # Collect (index, size, is_hbm) for each operand
        operands: list[tuple[int, int, bool]] = []
        for part in (p.strip() for p in payload.split(";;") if p.strip()):
            m = self._INLINED_CALL_OPERAND_INSTR_RE.match(part)
            if not m:
                continue
            rest = m.group(1)
            info = self._inlined_call_operand_info(rest)
            if info is None:
                continue
            base_address, size, index = info
            # Detect HBM vs scalar from the opcode (hbm vs <no memory space>)
            is_hbm = "inlined_call_operand.hbm" in part
            operands.append((index, size, is_hbm))

        # Assign base addresses: only HBM operands are stacked; <no memory space> excluded
        operands.sort(key=lambda x: x[0])
        hbm_offset = 0
        for index, size, is_hbm in operands:
            op_id = f"#operand{index}"
            if op_id not in symbol_table:
                if is_hbm:
                    base_address = hbm_offset
                    hbm_offset += size
                    space = "hbm"
                else:
                    base_address = 0  # scalar, not in HBM
                    space = "smem"
                symbol_table[op_id] = Allocation(
                    id=op_id, size=size, space=space, base_address=base_address
                )

    def parse_program(
        self, partial_path: Path
    ) -> tuple[dict[str, Allocation], dict[int, list[Instruction]]]:
        """Parse program from partial path; resolves #allocation refs to base addresses."""
        symbol_table = self._parse_symbol_table(partial_path)

        bundles: dict[int, list[Instruction]] = {}
        for addr, payload in self._iter_bundle_payloads(partial_path):
            self._collect_operands_into_symbol_table(payload, symbol_table)
            bundles[addr] = self.parse_bundle(payload, symbol_table)

        return symbol_table, bundles
