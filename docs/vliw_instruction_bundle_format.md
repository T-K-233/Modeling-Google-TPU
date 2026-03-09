# TPU VLIW / TLP Instruction Format Report

Generated: 2026-03-09

## Scope and Sources

This report merges both views of the same TPU instruction format:

1. Final executable bundle view:
   - `*-final_bundles.txt` corpus
2. TLP program view:
   - extracted `*_vliw_bundles.txt` corpus
   - includes thousands of TLP steps

Supporting artifacts:

- derived JSON aggregate statistics
- parser/analyzer scripts used to extract slot and operand patterns

## Unified Bundle Model

Bundle line shape:

`0xNN [optional control tag]: [optional >] { op ;; op ;; ... }`

Observed control tags:

- `LH`, `LB`, `LE`, `PB`, `PF`, `CT`

Inferred slot families:

| MXU slot 0 | MXU slot 1 | XLU slot 0 | XLU slot 1 | Vector ALU | Vector load channel(s) | Vector store channel(s) | Scalar / predicate | DMA / sync / branch |
|---|---|---|---|---|---|---|---|---|
| `*.mxu0` | `*.mxu1` | `*.xlu0` | `*.xlu1` | `vadd/vmul/vsel/...` | `vld*` | `vst*` | `s*`, `p*` | `dma.*`, `vsync*`, `sbr.rel` |

## Slot Mapping From `*-static-per-bundle-utilization.txt`

Hardware capacity (from per-bundle-utilization dumps):

`MXU=4 | XLU=3 | VALU=4 | EUP=1 | LOAD=3 | STORE=1 | SALU=2`

Inferred must-fit mapping:

| Utilization slot | Instruction forms that fit this slot |
|---|---|
| `MXU` | `*.mxu0`, `*.mxu1` (`vmatmul.*.mxu*`, `vmatprep.*.mxu*`, `vmatpush*.*.mxu*`, `vpop.f32.mrf.mxu*`) |
| `XLU` | `*.xlu0`, `*.xlu1` (`vpop.permute.xlu*`, `vrot.lane.*.xlu*`, `vbcast.lane.*.xlu*`, `vxpose.xlu*.*`) |
| `LOAD` | `vld*` (`vld`, `vld.sshfl`) |
| `STORE` | `vst*` (`vst`, `vst.msk`) |
| `EUP` | transcendental/special-function pop path: pre-delay/DGO `*.pop` ops and final-bundle `vpop.eup` |
| `VALU` | non-EUP vector compute (`vadd/vmul/vsub/vsel/vpack/...`) and producer side of transcendental ops (`vtanh.f32`, `vpow2.f32`, `vrcp.f32`, `vrsqrt.f32`, etc.) |
| `SALU` | scalar/predicate/control path (`s*`, `p*`, scalar addressing/pointer ops, branches, call setup, sync/control) |

Notes:

- Utilization files are pre-delay/final-hlo hardware-slot views; final bundle dumps may show additional compiler-introduced control/pseudo ops in the scalar/control stream.
- Observed slot maxima from utilization files match the published capacity vector exactly.

### Verification Pair A (TLP)

- Pre-delay utilization rows include `EUP=1` patterns such as:
  - `0 0 1 1 3 0 2`
  - `0 0 1 1 1 1 0`
- Corresponding packed bundles contain `vtanh.pop ...`
- Corresponding final bundles contain paired `vtanh.f32 ...` and `vpop.eup ...`

Interpretation: `vtanh.*` producer work is `VALU`, with pop/retire on `EUP`.

### Verification Pair B (TLP)

- Pre-delay/final-hlo utilization rows include `EUP=1` patterns such as:
  - `0 0 3 1 1 0 0`
  - `0 0 2 1 2 0 0`
- Corresponding packed bundles contain `vpow.pop ...`
- Corresponding final bundles contain paired `vpow2.f32 ...` and `vpop.eup ...`

Interpretation: the same producer/consumer split (`VALU` producer + `EUP` pop) holds on a second independent TLP program.

## Final-Bundle Quantitative Limits (`*-final_bundles.txt`)

- Files: 652
- Bundles: 800,128
- Max ops in one bundle line: 23
- Max distinct slot families in one bundle line: 8
  - `mxu_slot0`, `mxu_slot1`, `xlu_slot0`, `xlu_slot1`, `vector_alu`, `vector_load`, `vector_store`, `scalar`

Per-slot max ops in one bundle:

| Slot family | Max ops in one bundle | Example bundle ID | Example snippet |
|---|---:|---|---|
| `mxu_slot0` | 3 | `Bundle A` | `%v... = vpop.f32.mrf.mxu0 ;; %... = vmatprep.subr.mxu0 ... ;; %... = vmatprep.mubr.bf16.mxu0 ...` |
| `mxu_slot1` | 3 | `Bundle B` | `%... = vmatprep.mubr.bf16.mxu1 ... ;; %v... = vpop.f32.mrf.mxu1 ;; %... = vmatpush2.bf16.msra.mxu1 ...` |
| `xlu_slot0` | 2 | `Bundle C` | `%... = vbcast.lane.b32.xlu0 ... ;; ... ;; %v... = vpop.permute.xlu0 %...` |
| `xlu_slot1` | 2 | `Bundle D` | `%... = vbcast.lane.b32.xlu1 ... ;; ... ;; %v... = vpop.permute.xlu1 %...` |
| `vector_alu` | 5 | `Bundle E` | `%v... = vpop.eup ... ;; %v... = vadd.f32 ... ;; %v... = vor.u32 ... ;; %v... = vmul.f32 ... ;; %v... = vadd.f32 ...` |
| `vector_load` | 3 | `Bundle F` | `%v... = vld ... ;; ... ;; %v... = vld ... ;; %v... = vld ...` |
| `vector_store` | 2 | `Bundle G` | `%... = vst ... ;; %... = vst ... ;; %s... = sshll.u32 ...` |
| `scalar` | 15 | `Bundle H` | `%... = sdivrem.u32 ... ;; %s... = sshll.u32 ... ;; %s... = sphi ...` |
| `scalar_predicate` | 2 | `Bundle I` | `%p... = pneg ... ;; %p... = por ...` |
| `call_control` (`inlined_call_operand`) | 21 | `Bundle J` | `%s... = inlined_call_operand.vmem ... ;; ... (21 inlined_call_operand ops in this bundle)` |
| `dma` | 1 | `Bundle K` | `%... = dma.vmem_to_hbm [thread:$0] /*vmem=*/... /*size_in_granules=*/...` |
| `sync_control` | 1 | `Bundle L` | `%... = vsyncpa [#allocation...], ... ;; ...` |
| `branch_control` | 1 | `Bundle M` | `%... = sbr.rel (%p...) target bundleno = ...` |

Full non-truncated max-case bundle strings were used internally during extraction.

Bundle density:

- p50: 4 ops
- p90: 7 ops
- p95: 8 ops
- p99: 10 ops
- p99.9: 11 ops

Canonical mixed compute occupancy (when all major compute families co-occur):

| mxu0 | mxu1 | mxu2 | mxu3 | xlu0 | xlu1 | vector_alu | EUP unit | vector_load | vector_store | scalar ALU |
|-----:|-----:|-----:|-----:|-----:|-----:|-----------:|---------:|------------:|-------------:|-----------:|
|    1 |    1 |    1 |    1 |    1 |    1 |    up to 4 |        1 |     up to 3 |            1 |    up to 2 |

## TLP Program Fields That Also Belong In Format Documentation

TLP view adds program-structure fields not always explicit in final bundle files:

| Field | Example |
|---|---|
| TLP step metadata | `# TLP step: TLP-30 ...` |
| Step segmentation and local PC reuse | repeated `0`, `0x1`, ... per step/region |
| Symbolic branch target form | `sbr.rel (...) target = $region25` |
| Step section headers | `# ...-TLP-30-DGO-vliw-packed-bundles.txt` |

Observed counts from TLP extracts:

- TLP step headers: 2,848
- Symbolic region branches (`target = $region...`): 714

## Instruction Family Coverage Delta (TLP vs Final)

Opcode diff (`TLP - final`) shows program/runtime/control ops that should be documented in the unified format:

- `vtrace`, `vsettm`, `vdelay`, `sfence`
- `compiler-scheduling-barrier`
- `scalar_parameter_address`, `inlined_call`
- `dma.hbm_to_smem`
- `setrngseed`, `vrng`
- `vsetiar.raw.iar0`, `vsetiar.raw.iar1`

Examples with high TLP counts:

- `vtrace`: 46,180
- `compiler-scheduling-barrier`: 40,829
- `scalar_parameter_address`: 7,266
- `inlined_call`: 5,984

Opcode diff (`final - TLP`) also shows late-stage details:

- `sphi`, `shalt.err`, `scalar_lea.sflag`, `vld.sshfl`
- mask-specialized MXU families (`*.msk.*.mxu*`)

## Per-Slot Operand / Parameter Schema (Merged Final + TLP)

This section is per functional-unit slot, not per bundle.
Each operation is classified into exactly one source-operand combination and one parameter-key combination, so combinations are mutually exclusive categories.

Data source:

- merged corpus-level aggregate extraction
- generated by parser/analyzer tooling

Operand type legend:

- `vector_reg`: `%v...`
- `scalar_reg`: `%s...`
- `predicate_reg`: `%p...`
- `untyped_reg`: `%123`-style SSA temp
- `immediate_int` / `immediate_float` / `immediate_hex_*`: literal immediates
- `symbol_allocation`: `#allocation...`
- `memory_vmem` / `memory_smem`: explicit memory address-space operands
- `call_operand_*`: `inlined_call_operand.{vmem|hbm|<no memory space>}`
- Combination notation: `(a, b, c)` means one op has all three; repeated entries encode multiplicity, e.g. `(a, a, b)`

| Slot family | Destination type set | Source operand combinations (top observed) | Named parameter key combinations (top observed) | Representative opcodes |
|---|---|---|---|---|
| `mxu_slot0` | `dest_vector_reg` (196000), `dest_untyped_reg` (194932) | `(none)` (196000); `(vector_reg)` (194932) | `(none)` (390932) | `vpop.f32.mrf.mxu0`, `vmatprep.mubr.bf16.mxu0`, `vmatprep.subr.mxu0`, `vmatmul.mubr.bf16.gmra.mxu0`, `vmatpush1.bf16.msra.mxu0` |
| `mxu_slot1` | `dest_untyped_reg` (197804), `dest_vector_reg` (195094) | `(vector_reg)` (197804); `(none)` (195094) | `(none)` (392898) | `vpop.f32.mrf.mxu1`, `vmatprep.subr.mxu1`, `vmatprep.mubr.bf16.mxu1`, `vmatmul.mubr.bf16.gmra.mxu1`, `vmatpush1.bf16.msra.mxu1` |
| `xlu_slot0` | `dest_vector_reg` (36550), `dest_untyped_reg` (34104) | `(untyped_reg)` (19686); `(none)` (16864); `(immediate_int, immediate_int, immediate_int, vector_reg)` (14418); `(immediate_int, vector_reg)` (13709); `(scalar_reg, vector_reg)` (4413) | `(none)` (56236); `(vx, width)` (14418) | `vpop.permute.xlu0`, `vpop.trf.xlu0`, `vbcast.lane.b32.xlu0`, `vxpose.xlu0.b32.cont`, `vrot.lane.b32.xlu0` |
| `xlu_slot1` | `dest_vector_reg` (36130), `dest_untyped_reg` (33795) | `(untyped_reg)` (19694); `(none)` (16436); `(immediate_int, immediate_int, immediate_int, vector_reg)` (14101); `(immediate_int, vector_reg)` (13819); `(scalar_reg, vector_reg)` (4317) | `(none)` (55824); `(vx, width)` (14101) | `vpop.permute.xlu1`, `vpop.trf.xlu1`, `vbcast.lane.b32.xlu1`, `vxpose.xlu1.b32.cont`, `vrot.lane.b32.xlu1` |
| `vector_alu` | `dest_vector_reg` (3206482), `dest_untyped_reg` (97807), `none` (65330) | `(vector_reg, vector_reg)` (2108935); `(immediate_int, vector_reg)` (705648); `(immediate_float, immediate_int, vector_reg)` (251526); `(vector_reg)` (159808); `(untyped_reg)` (89714) | `(none)` (3189113); `(on_false_vx, on_true_vy, vm)` (178522); `(lhs_vy, low, rhs_vx)` (1984) | `vadd.f32`, `vmul.f32`, `vadd.s32`, `vxor.u32`, `vshrl.u32` |
| `vector_load` | `dest_vector_reg` (1218493) | `(immediate_hex_dollar, immediate_hex_dollar, memory_vmem, scalar_reg)` (701044); `(immediate_hex_dollar, immediate_hex_dollar, memory_vmem, symbol_allocation)` (284293); `(immediate_hex_dollar, memory_vmem, symbol_allocation)` (218568); `(immediate_hex_dollar, memory_vmem, scalar_reg)` (6458); `(immediate_hex_dollar, memory_vmem, scalar_reg, scalar_reg)` (5118) | `(sm, vmem)` (1073446); `(sm, ss, vmem)` (141601); `(pattern, sm, vmem)` (2978); `(sm, sps, ss, vmem)` (468) | `vld`, `vld.sshfl` |
| `vector_store` | `dest_untyped_reg` (730254) | `(immediate_hex_dollar, immediate_hex_dollar, memory_vmem, symbol_allocation, vector_reg)` (339730); `(immediate_hex_dollar, immediate_hex_dollar, memory_vmem, scalar_reg, vector_reg)` (241675); `(immediate_hex_dollar, memory_vmem, symbol_allocation, vector_reg)` (146764); `(immediate_hex_dollar, memory_vmem, scalar_reg, vector_reg)` (1829) | `(sm, vmem, vst_source)` (725650); `(sm, ss, vmem, vst_source)` (4568); `(sm, vm, vmem, vst_source)` (24); `(sm, ss, vm, vmem, vst_source)` (12) | `vst`, `vst.msk` |
| `scalar` | `dest_scalar_reg` (132850), `dest_predicate_reg` (29986), `dest_untyped_reg` (9572) | `(immediate_int, scalar_reg)` (45847); `(scalar_reg, scalar_reg)` (28632); `(memory_smem, symbol_allocation)` (23567); `(symbol_allocation)` (20825); `(immediate_int)` (12966); `(bool_flag, scalar_reg)` (11665); `(immediate_int, predicate_reg, scalar_reg)` (9283) | `(none)` (128003); `(smem)` (29775); `(resolvable)` (11665); `(on_false, on_true, predicate)` (1716); `(dst_syncflagno, hbm, hlo, size_in_granules, vmem)` (1079) | `smov`, `sld`, `scmp.lt.s32.totalorder`, `scalar_lea.vmem`, `scmp.ne.s32.totalorder` |
| `scalar_predicate` | `dest_predicate_reg` (14648) | `(predicate_reg, predicate_reg)` (14390); `(predicate_reg)` (258) | `(none)` (14648) | `pnand`, `por`, `pneg` |
| `dma` | `dest_untyped_reg` (14731) | `(immediate_int, symbol_allocation)` (7339); `(immediate_int, scalar_reg, scalar_reg, symbol_allocation, symbol_thread)` (5448); `(immediate_int, scalar_reg, scalar_reg, scalar_reg, scalar_reg, scalar_reg, symbol_allocation, symbol_thread)` (761); `(scalar_reg, scalar_reg, scalar_reg, symbol_allocation, symbol_thread)` (376); `(immediate_int, scalar_reg, scalar_reg, symbol_allocation)` (192) | `(none)` (7684); `(dst_syncflagno, hbm, size_in_granules, thread, vmem)` (6005); `(dst_stride, dst_syncflagno, hbm, size_in_granules, src_stride, steps_per_stride, thread, vmem)` (850); `(dst_syncflagno, hbm, size_in_granules, smem)` (192) | `dma.done.wait`, `dma.hbm_to_vmem`, `dma.vmem_to_hbm`, `dma.done`, `dma.hbm_to_smem` |
| `sync_control` | `dest_untyped_reg` (37087) | `(immediate_int, symbol_allocation)` (22228); `(scalar_reg)` (5696); `(immediate_int)` (3204); `(none)` (3040); `(immediate_hex_dollar, immediate_int, symbol_allocation)` (2226) | `(none)` (37087) | `vsyncpa`, `vsyncadd`, `vsettm`, `sfence`, `vdelay` |
| `branch_control` | `dest_untyped_reg` (6157) | `(immediate_hex_plain, immediate_int, immediate_int, predicate_reg)` (5443); `(predicate_reg, symbol_region)` (714) | `(region, target_bundleno)` (5443); `(target)` (714) | `sbr.rel` |
| `call_control` | `dest_scalar_reg` (12879), `dest_untyped_reg` (5984) | `(immediate_int)` (7266); `(scalar_reg, scalar_reg)` (2208); `(scalar_reg, scalar_reg, scalar_reg)` (2144); `(call_operand_vmem, immediate_int, immediate_int)` (816); `(call_operand_hbm, immediate_int, immediate_int, immediate_int, immediate_int)` (815); `(call_operand_vmem, immediate_int, immediate_int, immediate_int)` (781); `(call_operand_no_space, immediate_int)` (730) | `(none)` (13250); `(index, kind, shape, shape_index)` (5503); `(alias, index, kind, shape, shape_index)` (110) | `scalar_parameter_address`, `inlined_call`, `inlined_call_operand` |
| `trace_debug` (TLP-only) | `dest_untyped_reg` (112806) | `(none)` (66626); `(immediate_int)` (34788); `(scalar_reg)` (11392) | `(none)` (112806) | `compiler-scheduling-barrier`, `vtrace` |

Example exclusivity reading:

- `vector_alu` observed many `(vector_reg, vector_reg)` and `(immediate_int, vector_reg)` forms, but no `(immediate_int, immediate_int)`-only top pattern.
