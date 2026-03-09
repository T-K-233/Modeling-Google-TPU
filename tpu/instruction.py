from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class Operation:
    mnemonic: str
    apply_effect: Callable


class IsaSpec:
    operations: dict[str, Operation] = {}


@dataclass
class InstructionBundle:
    """
    TPU v5e-1 has the following instruction bundle slots:
      MXU|  XLU| VALU|  EUP| LOAD|STORE| SALU
      4     3     4     1     3     1     2
    """
    address: int
    instructions: list[Any] = field(default_factory=list)
    mxu0: dict = field(default_factory=dict)
    mxu1: dict = field(default_factory=dict)
    mxu2: dict = field(default_factory=dict)
    mxu3: dict = field(default_factory=dict)
    xlu0: dict = field(default_factory=dict)
    xlu1: dict = field(default_factory=dict)
    xlu2: dict = field(default_factory=dict)
    valu0: dict = field(default_factory=dict)
    valu1: dict = field(default_factory=dict)
    valu2: dict = field(default_factory=dict)
    valu3: dict = field(default_factory=dict)
    eup: dict = field(default_factory=dict)
    load0: dict = field(default_factory=dict)
    load1: dict = field(default_factory=dict)
    load2: dict = field(default_factory=dict)
    store0: dict = field(default_factory=dict)
    salu0: dict = field(default_factory=dict)
    salu1: dict = field(default_factory=dict)

    # def __str__(self):
    #     breakpoint()
    #     mxu0_slot = self.mxu0.get("opcode") if self.mxu0 else "nop"
    #     mxu1_slot = self.mxu1.get("opcode") if self.mxu1 else "nop"
    #     mxu2_slot = self.mxu2.get("opcode") if self.mxu2 else "nop"
    #     mxu3_slot = self.mxu3.get("opcode") if self.mxu3 else "nop"
    #     xlu0_slot = self.xlu0.get("opcode") if self.xlu0 else "nop"
    #     xlu1_slot = self.xlu1.get("opcode") if self.xlu1 else "nop"
    #     xlu2_slot = self.xlu2.get("opcode") if self.xlu2 else "nop"
    #     valu0_slot = self.valu0.get("opcode") if self.valu0 else "nop"
    #     valu1_slot = self.valu1.get("opcode") if self.valu1 else "nop"
    #     valu2_slot = self.valu2.get("opcode") if self.valu2 else "nop"
    #     valu3_slot = self.valu3.get("opcode") if self.valu3 else "nop"
    #     eup_slot = self.eup.get("opcode") if self.eup else "nop"
    #     load0_slot = self.load0.get("opcode") if self.load0 else "nop"
    #     load1_slot = self.load1.get("opcode") if self.load1 else "nop"
    #     load2_slot = self.load2.get("opcode") if self.load2 else "nop"
    #     store0_slot = self.store0.get("opcode") if self.store0 else "nop"
    #     salu0_slot = self.salu0.get("opcode") if self.salu0 else "nop"
    #     salu1_slot = self.salu1.get("opcode") if self.salu1 else "nop"
    #     info = f"0x{self.address:x}: | {mxu0_slot} | {mxu1_slot} | {mxu2_slot} | {mxu3_slot} | {xlu0_slot} | {xlu1_slot} | {xlu2_slot} | {valu0_slot} | {valu1_slot} | {valu2_slot} | {valu3_slot} | {eup_slot} | {load0_slot} | {load1_slot} | {load2_slot} | {store0_slot} | {salu0_slot} | {salu1_slot} |"
    #     return info


def instr(mnemonic):
    if not isinstance(mnemonic, str):
        raise TypeError("@instr decorator must be @instr(<your instruction>)")

    def effect(func: Callable) -> Callable:
        IsaSpec.operations[mnemonic] = Operation(
            mnemonic=mnemonic,
            apply_effect=func,
        )

        return func

    return effect
