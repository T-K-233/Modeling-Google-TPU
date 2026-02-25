import torch


def convert_to_bf16_tile_layout(data: torch.Tensor, num_sublanes: int, num_lanes: int) -> torch.Tensor:
    """
    Convert BF16 contiguous data to a tiled format.

    Args:
        data: The data to pack.
        num_lanes: The number of lanes.
        num_sublanes: The number of sublanes.

        Input:
        [
            [ 0,  1,  2,  3,  4,  5,  6,  7, ]
            [ 8,  9, 10, 11, 12, 13, 14, 15, ]
            [16, 17, 18, 19, 20, 21, 22, 23, ]
            [24, 25, 26, 27, 28, 29, 30, 31, ]
        ]
        Output:
        [
            [ 0,  2,  4,  6,  8, 10, 12, 14, ]
            [ 1,  3,  5,  7,  9, 11, 13, 15, ]
            [16, 18, 20, 22, 24, 26, 28, 30, ]
            [17, 19, 21, 23, 25, 27, 29, 31, ]
        ]

    """
    num_elements_per_row = num_lanes * 2
    return data \
        .reshape(-1, num_elements_per_row, 2) \
        .permute(0, 2, 1) \
        .reshape(-1, num_elements_per_row) \
        .contiguous()


def convert_from_bf16_tile_layout(data: torch.Tensor, num_sublanes: int, num_lanes: int) -> torch.Tensor:
    """
    Convert BF16 tiled data to contiguous format.

    Args:
        data: The data to unpack.
        num_lanes: The number of lanes.
        num_sublanes: The number of sublanes.

        Input:
        [
            [ 0,  2,  4,  6,  8, 10, 12, 14, ]
            [ 1,  3,  5,  7,  9, 11, 13, 15, ]
            [16, 18, 20, 22, 24, 26, 28, 30, ]
            [17, 19, 21, 23, 25, 27, 29, 31, ]
        ]
        Output:
        [
            [ 0,  1,  2,  3,  4,  5,  6,  7, ]
            [ 8,  9, 10, 11, 12, 13, 14, 15, ]
            [16, 17, 18, 19, 20, 21, 22, 23, ]
            [24, 25, 26, 27, 28, 29, 30, 31, ]
        ]
    """
    num_elements_per_row = num_lanes * 2
    return data \
        .reshape(-1, 2, num_elements_per_row) \
        .permute(0, 2, 1) \
        .reshape(-1, num_elements_per_row) \
        .contiguous()
