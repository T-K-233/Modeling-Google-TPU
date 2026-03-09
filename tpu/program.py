from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .instruction import InstructionBundle


@dataclass
class KernelProgram:
    name: str
    stem: Path
    symbol_table: dict
    bundles: dict[int, InstructionBundle]
    kind: str
    numeric_id: int | None = None
    kernel_name: str | None = None
    hlo_param_shapes: list[str] = field(default_factory=list)
    hlo_output_shape: str | None = None
    hlo_output_nbytes: int = 0
    operand_count: int = 0
    output_operand_indices: list[int] = field(default_factory=list)
    output_operand_nbytes: list[int] = field(default_factory=list)
    operand_descriptors: dict[int, dict[str, object]] = field(default_factory=dict)
    start_pc: int = 0


@dataclass
class Program:
    kernels: dict[str, KernelProgram] = field(default_factory=dict)
    mode: str = "single"
    source_dir: Path | None = None
    entry_program_name: str = ""
    tlp_order: list[str] = field(default_factory=list)
    kernel_candidates_by_name: dict[str, list[str]] = field(default_factory=dict)
    initialized_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
