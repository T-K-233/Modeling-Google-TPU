import torch


def convert_to_bf16_tile_layout(data: torch.Tensor, num_sublanes: int, num_lanes: int) -> torch.Tensor:
    """
    Convert logical BF16 matrix [num_sublanes, num_lanes] to TPU BF16 tiled layout.

    TPU BF16 uses repeated tiling (8,128)(2,1), i.e. values are paired across
    the second-minor dimension (sublanes) at the same lane:
      packed[r, c, 0] = logical[2*r, c]
      packed[r, c, 1] = logical[2*r+1, c]

    Returns shape [num_sublanes // 2, num_lanes * 2] (equivalent to
    [num_sublanes // 2, num_lanes, 2] flattened across the most-minor dims).
    """
    half_sublanes = num_sublanes // 2

    if data.ndim == 3:
        # Backward compatibility for callers already materializing [half, lanes, 2].
        assert data.shape == (half_sublanes, num_lanes, 2), (
            f"Expected data shape {(half_sublanes, num_lanes, 2)}, got {tuple(data.shape)}"
        )
        return data.reshape(half_sublanes, num_lanes * 2).contiguous()

    assert data.shape == (num_sublanes, num_lanes), (
        f"Expected data shape {(num_sublanes, num_lanes)}, got {tuple(data.shape)}"
    )
    even_rows = data[0::2, :]
    odd_rows = data[1::2, :]
    return torch.stack([even_rows, odd_rows], dim=-1).reshape(half_sublanes, num_lanes * 2).contiguous()


def convert_from_bf16_tile_layout(data: torch.Tensor, num_sublanes: int, num_lanes: int) -> torch.Tensor:
    """
    Convert TPU BF16 tiled layout back to logical [num_sublanes, num_lanes].
    """
    half_sublanes = num_sublanes // 2
    if data.ndim == 1:
        assert data.numel() == half_sublanes * num_lanes * 2, (
            f"Expected {half_sublanes * num_lanes * 2} elements, got {data.numel()}"
        )
        data = data.reshape(half_sublanes, num_lanes * 2)
    elif data.ndim == 3:
        assert data.shape == (half_sublanes, num_lanes, 2), (
            f"Expected data shape {(half_sublanes, num_lanes, 2)}, got {tuple(data.shape)}"
        )
        data = data.reshape(half_sublanes, num_lanes * 2)
    else:
        assert data.shape == (half_sublanes, num_lanes * 2), (
            f"Expected data shape {(half_sublanes, num_lanes * 2)}, got {tuple(data.shape)}"
        )

    paired = data.reshape(half_sublanes, num_lanes, 2)
    result = torch.empty((num_sublanes, num_lanes), dtype=data.dtype, device=data.device)
    result[0::2, :] = paired[:, :, 0]
    result[1::2, :] = paired[:, :, 1]
    return result.contiguous()


def pack_bf16_register(low: torch.Tensor, high: torch.Tensor, num_sublanes: int, num_lanes: int) -> torch.Tensor:
    """
    Pack two BF16 logical matrices into one TPU BF16 vector register image [8, 256].

    Register rows [0:half) carry low, rows [half:8) carry high.
    """
    assert low.shape == (num_sublanes, num_lanes), (
        f"Expected low shape {(num_sublanes, num_lanes)}, got {tuple(low.shape)}"
    )
    assert high.shape == (num_sublanes, num_lanes), (
        f"Expected high shape {(num_sublanes, num_lanes)}, got {tuple(high.shape)}"
    )
    half_sublanes = num_sublanes // 2
    packed = torch.empty((num_sublanes, num_lanes * 2), dtype=low.dtype, device=low.device)
    packed[:half_sublanes, :] = convert_to_bf16_tile_layout(low, num_sublanes, num_lanes)
    packed[half_sublanes:, :] = convert_to_bf16_tile_layout(high, num_sublanes, num_lanes)
    return packed.contiguous()


def unpack_bf16_register(data: torch.Tensor, num_sublanes: int, num_lanes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack one TPU BF16 vector register image [8, 256] into (low, high) logical matrices.
    """
    assert data.shape == (num_sublanes, num_lanes * 2), (
        f"Expected register shape {(num_sublanes, num_lanes * 2)}, got {tuple(data.shape)}"
    )
    half_sublanes = num_sublanes // 2
    low = convert_from_bf16_tile_layout(data[:half_sublanes, :], num_sublanes, num_lanes)
    high = convert_from_bf16_tile_layout(data[half_sublanes:, :], num_sublanes, num_lanes)
    return low, high


def pack_bf16_mxu_rhs_coalesced(weight: torch.Tensor) -> torch.Tensor:
    """
    Pack logical BF16 MXU RHS [128, 8] into the coalesced VMEM image [8, 256].

    This matches the load path used by kernels that emit:
      vld (coalesced, +offset) -> vunpack.c.[l|h].bf16 -> vmatpush.msra.mxu*

    The produced layout occupies one full 4096-byte register image.
    """
    assert weight.shape == (128, 8), f"Expected weight shape (128, 8), got {tuple(weight.shape)}"

    out = torch.zeros((8, 256), dtype=weight.dtype, device=weight.device)
    flat = out.reshape(-1)

    row = torch.arange(128, device=weight.device)
    row_pair = row // 2
    major = row_pair % 8
    minor = row_pair // 8
    base = major * 256 + minor * 16 + (row % 2)
    col_offsets = (2 * torch.arange(8, device=weight.device)).unsqueeze(0)
    dst = base.unsqueeze(1) + col_offsets
    flat[dst] = weight
    return out.contiguous()
