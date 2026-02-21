from typing import Callable
from dataclasses import dataclass


@dataclass
class Operation:
    mnemonic: str
    apply_effect: Callable


class IsaSpec:
    operations: dict[str, Operation] = {}


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
