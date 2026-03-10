from enum import Enum
from typing import Callable
from dataclasses import dataclass, field


class BundleSlotType(Enum):
    MXU = "MXU"
    XLU = "XLU"
    VALU = "VALU"
    EUP = "EUP"
    LOAD = "LOAD"
    STORE = "STORE"
    SALU = "SALU"


@dataclass
class Operation:
    mnemonic: str
    slot_type: BundleSlotType
    apply_effect: Callable


class IsaSpec:
    operations: dict[str, Operation] = {}


OperandValue = int | float | str
AddressOperand = int | str | tuple[str, int]


@dataclass
class MXUSlotParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    vs1_reg: str = ""
    vm_reg: str = ""


@dataclass
class XLUSlotParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    vs1_reg: str = 0
    rs1_reg: str = 0
    immediate: int = 0
    immediate2: int = 0


@dataclass
class VALUSlotParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    vs1_reg: str = 0
    vs2_reg: str = 0
    vm_reg: str = ""
    vs1_imm: int = 0
    vs2_imm: int = 0


@dataclass
class EUPParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    vs1_reg: str = ""


@dataclass
class LOADParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    address: AddressOperand = 0
    sublane_mask: int = 255
    sublane_stride: int | None = 0


@dataclass
class STOREParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    address: AddressOperand = 0
    sublane_mask: int = 255
    sublane_stride: int = 0
    vs1_reg: str = ""
    vm_reg: str = ""


@dataclass
class SALUSlotParams:
    opcode: str = ""
    valid: bool = False
    vd_reg: str = ""
    order: int = 0
    pred: str | None = None
    predication: bool = False
    has_imm1: bool = False
    has_rs1: bool = False
    immediate: int = 0
    immediate2: int = 0
    rs1_reg: str = 0
    rs2_reg: str = 0
    address: AddressOperand = 0
    sync_flag: int = 0
    ps1_reg: str = ""
    ps2_reg: str = ""
    target: int = 0
    call_args: list[str | int] = field(default_factory=list)
    callee: str = ""
    kernel_program: str = ""
    operand_index: int = -1
    operand_kind: str = ""
    operand_dtype: str = ""
    operand_dims: list[int] = field(default_factory=list)


SlotParams = (
    MXUSlotParams
    | XLUSlotParams
    | VALUSlotParams
    | EUPParams
    | LOADParams
    | STOREParams
    | SALUSlotParams
)


@dataclass
class InstructionBundle:
    """
    TPU v5e-1 has the following instruction bundle slots:
      MXU|  XLU| VALU|  EUP| LOAD|STORE| SALU
      4     3     4     1     3     1     2
    """
    address: int
    mxu0: MXUSlotParams = field(default_factory=MXUSlotParams)
    mxu1: MXUSlotParams = field(default_factory=MXUSlotParams)
    mxu2: MXUSlotParams = field(default_factory=MXUSlotParams)
    mxu3: MXUSlotParams = field(default_factory=MXUSlotParams)
    xlu0: XLUSlotParams = field(default_factory=XLUSlotParams)
    xlu1: XLUSlotParams = field(default_factory=XLUSlotParams)
    xlu2: XLUSlotParams = field(default_factory=XLUSlotParams)
    valu0: VALUSlotParams = field(default_factory=VALUSlotParams)
    valu1: VALUSlotParams = field(default_factory=VALUSlotParams)
    valu2: VALUSlotParams = field(default_factory=VALUSlotParams)
    valu3: VALUSlotParams = field(default_factory=VALUSlotParams)
    eup: EUPParams = field(default_factory=EUPParams)
    load0: LOADParams = field(default_factory=LOADParams)
    load1: LOADParams = field(default_factory=LOADParams)
    load2: LOADParams = field(default_factory=LOADParams)
    store0: STOREParams = field(default_factory=STOREParams)
    salu0: SALUSlotParams = field(default_factory=SALUSlotParams)
    salu1: SALUSlotParams = field(default_factory=SALUSlotParams)
    overflow_slots: list[SlotParams] = field(default_factory=list, repr=False)

    _TABLE_COLUMNS = [
        ("MXU", ["mxu0", "mxu1", "mxu2", "mxu3"]),
        ("XLU", ["xlu0", "xlu1", "xlu2"]),
        ("VALU", ["valu0", "valu1", "valu2", "valu3"]),
        ("EUP", ["eup"]),
        ("LOAD", ["load0", "load1", "load2"]),
        ("STORE", ["store0"]),
        ("SALU", ["salu0", "salu1"]),
    ]
    _TABLE_COL_WIDTH = 16

    @classmethod
    def table_divider(cls) -> str:
        header = " " * 6 + " | " + " | ".join(
            f"{name:^{cls._TABLE_COL_WIDTH}}" for name, _ in cls._TABLE_COLUMNS
        )
        return "-" * len(header)

    @staticmethod
    def _slot_column_name(slot: SlotParams) -> str:
        if isinstance(slot, MXUSlotParams):
            return "MXU"
        if isinstance(slot, XLUSlotParams):
            return "XLU"
        if isinstance(slot, VALUSlotParams):
            return "VALU"
        if isinstance(slot, EUPParams):
            return "EUP"
        if isinstance(slot, LOADParams):
            return "LOAD"
        if isinstance(slot, STOREParams):
            return "STORE"
        return "SALU"

    def iter_valid_slots(self) -> list[SlotParams]:
        slots: list[SlotParams] = []
        for _, slot_names in self._TABLE_COLUMNS:
            slots.extend(getattr(self, slot_name) for slot_name in slot_names)
        slots.extend(self.overflow_slots)
        valid_slots = [slot for slot in slots if slot.valid]
        return sorted(valid_slots, key=lambda slot: int(slot.order))

    def table_repr(self, include_header: bool = True) -> str:
        columns: list[tuple[str, list[SlotParams]]] = []
        for name, slot_names in self._TABLE_COLUMNS:
            column_slots = [getattr(self, slot_name) for slot_name in slot_names if getattr(self, slot_name).valid]
            columns.append((name, column_slots))
        for slot in self.overflow_slots:
            if not slot.valid:
                continue
            column_name = self._slot_column_name(slot)
            for name, column_slots in columns:
                if name == column_name:
                    column_slots.append(slot)
                    break
        max_rows = max((len(column_slots) for _, column_slots in columns), default=0)
        col_width = self._TABLE_COL_WIDTH

        def cell(column_slots: list[SlotParams], row: int) -> str:
            if row >= len(column_slots):
                return ""
            slot = column_slots[row]
            return slot.opcode if (slot.valid and slot.opcode) else ""

        lines: list[str] = []
        if include_header:
            header = " " * 6 + " | " + " | ".join(f"{name:^{col_width}}" for name, _ in self._TABLE_COLUMNS)
            sep = "-" * len(header)
            lines = [header, sep]
        first_data_row = True
        for row_idx in range(max_rows):
            cells = [cell(column_slots, row_idx) for _, column_slots in columns]
            if any(cells):
                prefix = f"0x{self.address:04X}" if first_data_row else " " * 6
                first_data_row = False
                lines.append(prefix + " | " + " | ".join(f"{c:^{col_width}}" for c in cells))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.table_repr(include_header=True)


def instr(mnemonic: str, slot: BundleSlotType) -> Callable:
    assert isinstance(mnemonic, str), "@instr decorator must be @instr(<instruction mnemonic>)"

    def effect(func: Callable) -> Callable:
        assert callable(func), "@instr decorator must be @instr(<instruction mnemonic>)"

        def apply_effect(state, params: SlotParams):
            return func(state, params)

        IsaSpec.operations[mnemonic] = Operation(
            mnemonic=mnemonic,
            slot_type=slot,
            apply_effect=apply_effect,
        )

        return func

    return effect
