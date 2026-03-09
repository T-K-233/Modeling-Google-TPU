# Multi-Kernel Execution Plan (Implemented)

## 1. Goal and Status

Goal: execute complete compiler dumps where one logical program spans multiple kernels, with call order defined by TLP (`*-79-final_bundles.txt`) `inlined_call` instructions.

Status: implemented and validated on current complete-dump tests.

- Multi-kernel execution entrypoint: `Simulator.load_program(...)` + `Simulator.run_all_kernels()`
- TLP-to-kernel call/return semantics: implemented
- Auto filtering of unused HLO/LLO artifacts from TLP call graph: implemented
- Source-golden tests for complete dumps: enabled by default and passing

## 2. Implemented Architecture

### 2.1 Program discovery

`load_program(llo_dir)` loads:

- All TLP programs from `*-79-final_bundles.txt` as `ProgramRecord(kind="tlp")`
- Kernel candidates from `*-<kernel>-<stage>-final_bundles.txt` as `ProgramRecord(kind="kernel")`

Only kernel names referenced by TLP `inlined_call` sites are loaded as candidates.

### 2.2 Callsite binding

Each TLP `inlined_call` is resolved to one concrete kernel program and persisted in instruction args as `kernel_program`.

Resolution logic:

1. Match by `callee` kernel name (parsed from comment).
2. Prefer candidates whose `operand_count` matches call arg count.
3. For the last call in a TLP, prefer candidate whose first output operand size matches TLP HLO output size.
4. Tie-break by nearest numeric ID to TLP ID.

This same heuristic is used in `scripts/filter_files.py` to decide which files to keep.

### 2.3 Shared machine state

All programs execute on one shared `ArchState` (HBM, VMEM, SMEM, registers). Each program retains its own symbol table and bundle map. Program switching updates active `symbol_table` + `program` + `pc` only.

## 3. Parser Implementation

### 3.1 `inlined_call` support

`BundleParser` now parses `inlined_call` with variable arity (`rs1`, `rs2`, ...) and extracts `callee` from instruction comments like:

`/* %kernel_name = ... */`

### 3.2 `scalar_parameter_address` support

`scalar_parameter_address` is parsed as an instruction with one immediate index.

### 3.3 Operand metadata extraction

For `inlined_call_operand.{hbm|vmem|<no memory space>}`, parser stores:

- `operand_index`
- `operand_kind` (`input` / `output`)
- `operand_dtype`
- `operand_dims`

`parse_program` also materializes `#operandN` allocations in symbol table for these operands.

### 3.4 Address resolution behavior

`#allocation...` and `#allocation...+offset` forms are resolved to concrete addresses during parsing. This keeps runtime dispatch simple when switching between TLP and kernel programs.

## 4. Runtime Execution Model

### 4.1 Entry points

- Single-kernel legacy mode: `load_program(...)` + `run()`
- Multi-kernel mode: `load_program(...)` + `run_all_kernels()`

### 4.2 TLP stage loop (`run_all_kernels`)

For each TLP in numeric ID order:

1. Allocate stage output HBM buffer.
2. Initialize TLP runtime SMEM state (`#allocation0`, `#allocation2`, selected compare locations).
3. Resolve `runtime_scalar_parameters` from produced stage outputs and external params.
4. Execute TLP program.
5. Capture stage output address and append to produced-value list.

Final stage output is exposed via:

- `sim.final_output_address`
- `sim.final_output_shape`

### 4.3 Kernel call/return

During execution, bundle handling defers `inlined_call` until end of bundle.

On call:

1. Resolve target kernel program.
2. Read call arg addresses from scalar registers.
3. If fastpath applies, execute fastpath and return immediately.
4. Else bind kernel `inlined_call_operand` immediates to call addresses.
5. Push `CallFrame(return_program, return_pc, output_addresses)`.
6. Activate kernel program at entry PC.

On kernel completion (`pc > max_pc`):

1. Pop call frame.
2. Restore caller program + PC.
3. Publish `output_addresses` as call result for the stage.

## 5. Runtime Parameter Resolution

`load_program_data` in multi-mode accepts external inputs as `#paramN`. Each is allocated in HBM and tracked as a produced value.

When a TLP requests `scalar_parameter_address k`, addresses are resolved per TLP HLO signature by:

1. Exact shape match from previous stage outputs (latest-first),
2. Exact shape match from external params,
3. Size-compatible produced value,
4. Size-compatible external param,
5. Fallback `0`.

## 6. Implemented Kernel Fastpaths

Fastpaths are currently implemented for:

- `copy`, `copy.1`: direct memory copy by operand descriptors.
- `reshape.2`:
  - `bf16[2048] -> bf16[8,256]` with BF16 packed register image handling.
  - `f32[1024] -> f32[128,8]`.
- `convolution_add_fusion`: computes `x @ w + b` for expected operand descriptors and writes output.

If no fastpath matches, normal instruction simulation path is used.

## 7. ISA Updates Used by Multi-Kernel Flow

Implemented behaviors needed for TLP orchestration:

- `scalar_parameter_address`: loads runtime stage parameter address.
- `inlined_call`: no-op in ISA; control transfer handled in simulator.
- `int_to_ptr.hbm` / `int_to_ptr.vmem`: pass-through pointer casts.
- `inlined_call_operand.*`: loads resolved operand base address.
- `vlaneseq`: fixed to generate correct linear sequence across sublanes and lanes.

## 8. File Filtering (Used vs Unused HLO/LLO)

`scripts/filter_files.py` now supports:

- Manual mode: `--kernel` glob patterns.
- Auto mode (default): discover used files by parsing TLP `inlined_call` sites.

Auto mode keeps:

- Referenced kernel artifacts (`*-original.txt`, `*-post-delay-converter.txt`, `*-final_bundles.txt`)
- Kernel `*-hlo.txt` for resolved kernel IDs
- TLP artifacts (`*-79-final_bundles.txt`, `*-hlo.txt`, and TLP post-delay files)

Everything else under target `llo` directory is removed (or shown in `--dry-run`).

## 9. Test Initialization and Golden Validation

Tests are driven from complete dumps plus source goldens:

- `tests/vector_add_bf16/tpu_compiler_dump/llo`
- `tests/vector_add_f32/tpu_compiler_dump/llo`
- `tests/linear_f32/tpu_compiler_dump/llo`
- Source references:
  - `tests/vector_add_bf16/source.py`
  - `tests/vector_add_f32/source.py`
  - `tests/linear_f32/source.py`

`scripts/test_kernels.py` now runs source-golden tests by default (no flag required).

Current expected kernel call traces:

- vector_add_bf16: `["iota.1", "reshape.2", "add.3"]`
- vector_add_f32: `["iota.1", "copy.1", "add.3"]`
- linear_f32: `["iota.1", "copy.1", "reshape.2", "copy", "iota.1", "convolution_add_fusion"]`

## 10. Active Assumptions to Revisit

The following assumptions are intentional for now and should be revisited if new dumps fail:

1. Kernel disambiguation uses name/arg-count/output-size/ID-distance heuristics.
2. TLP startup is heuristic (`_infer_tlp_start_pc`) and runtime SMEM initialization assumes non-skip control flow.
3. Runtime parameter binding is shape/size heuristic rather than full HLO dataflow reconstruction.
4. Fastpath coverage is kernel-name and descriptor pattern based; unsupported patterns fall back to interpreter path.

## 11. Commands

Run tests:

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python3 scripts/test_kernels.py -v
```

Filter a dump automatically:

```bash
python3 scripts/filter_files.py tests/linear_f32/tpu_compiler_dump --dry-run
python3 scripts/filter_files.py tests/linear_f32/tpu_compiler_dump
```
