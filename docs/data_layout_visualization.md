# Data Layout Visualization

This page provides quick visual references for the current BF16 layouts modeled in this repository.

## ASCII (quick mental model)

```text
BF16 tile packing toy example
logical (8x8) -> tiled (4x8x2) -> flattened (4x16)

logical rows (r0..r7):
  r0: 00 01 02 03 04 05 06 07
  r1: 08 09 10 11 12 13 14 15
  r2: 16 17 18 19 20 21 22 23
  r3: 24 25 26 27 28 29 30 31
  r4: 32 33 34 35 36 37 38 39
  r5: 40 41 42 43 44 45 46 47
  r6: 48 49 50 51 52 53 54 55
  r7: 56 57 58 59 60 61 62 63

flattened packed rows (pair even/odd in minor dim):
  p0: 00 08 01 09 02 10 03 11 04 12 05 13 06 14 07 15
  p1: 16 24 17 25 18 26 19 27 20 28 21 29 22 30 23 31
  p2: 32 40 33 41 34 42 35 43 36 44 37 45 38 46 39 47
  p3: 48 56 49 57 50 58 51 59 52 60 53 61 54 62 55 63
```

This is the `(8,128)(2,1)` behavior in miniature.

## Generate Visual Figures (grid heatmaps)

Run:

```bash
python scripts/visualize_layouts.py
```

The script writes two PNG files at the project root:

1. `toy_layouts.png`
1. `actual_layouts.png`

`toy_layouts.png` includes:

1. Toy logical grid (`8x8`)
1. Toy tiled view component 0 (`4x8`, even-row component)
1. Toy tiled view component 1 (`4x8`, odd-row component)
1. Toy flattened packed grid (`4x16`)

`actual_layouts.png` includes:

1. Memory/Register physical image for BF16 tile layout (`8x128 -> 4x256`)
1. Register physical image for `vpack.c.bf16` (`8x256`, low/high halves)
1. Memory physical image for MXU RHS coalesced layout (`128x8 -> 8x256`, unused cells highlighted)

## Related Docs

- [bf16_tiled_layout.md](./bf16_tiled_layout.md)
- [bf16_mxu_rhs_coalesced_layout.md](./bf16_mxu_rhs_coalesced_layout.md)
