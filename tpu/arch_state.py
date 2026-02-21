import torch

from .memory import Memory


class ArchState:
    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        self.pc = 0
        self.next_pc = 0
        self.hbm_size = 1024 * (1 << 20)  # 1024 MiB
        self.vmem_size = 128 * (1 << 20)  # 128 MiB
        self.smem_size = 1 * (1 << 20)  # 1 MiB
        self.sflag_size = 256  # 256 bytes (64 words)
        self.num_x_registers = 64
        self.num_v_registers = 64

        self.num_lanes = 128
        self.num_sublanes = 8
        self.vreg_size = self.num_lanes * self.num_sublanes * torch.float32.itemsize

        self.initialize_buffers()

    def initialize_buffers(self) -> None:
        self.hbm = Memory("HBM", self.hbm_size, verbose=self.verbose)
        self.vmem = Memory("VMEM", self.vmem_size, verbose=self.verbose)
        self.smem = Memory("SMEM", self.smem_size, verbose=self.verbose)
        self.sflag = Memory("SFLAG", self.sflag_size, verbose=self.verbose)

        self.xreg: dict[str, int] = {}
        for index in range(self.num_x_registers):
            self.xreg[f"s{index}"] = index

        self.vreg: dict[str, torch.Tensor] = {}
        for index in range(self.num_v_registers):
            self.vreg[f"v{index}"] = torch.zeros(
                self.vreg_size,
                dtype=torch.uint8,
            )

        self.mxu0_weight_buffer = torch.zeros(self.num_lanes, self.num_lanes, dtype=torch.float32)
        self.mxu0_accumulator = torch.zeros(self.num_lanes, self.num_lanes, dtype=torch.float32)

    def read_xreg(self, src: str) -> int:
        return self.xreg[src]

    def write_xreg(self, dest: str, value: int) -> None:
        self.xreg[dest] = value

    def read_vreg(self, src: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (
            self.vreg[src]
            .view(dtype)
            .reshape(self.num_sublanes, -1)
        )

    def write_vreg(self, dest: str, value: torch.Tensor) -> None:
        assert value.nbytes == self.vreg_size, f"Value size mismatch: {value.nbytes} != {self.vreg_size}"
        self.vreg[dest].view(value.dtype)[:] = value.flatten()

    def read_hbm(self, address: int, size: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return self.hbm.read(address, size).view(dtype)

    def write_hbm(self, address: int, data: torch.Tensor) -> None:
        self.hbm.write(address, data.flatten().view(torch.uint8))

    def read_vmem(self, address: int, size: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return self.vmem.read(address, size).view(dtype)

    def write_vmem(self, address: int, data: torch.Tensor) -> None:
        self.vmem.write(address, data.flatten().view(torch.uint8))

    def read_smem(self, address: int, size: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return self.smem.read(address, size).view(dtype)

    def write_smem(self, address: int, data: torch.Tensor) -> None:
        self.smem.write(address, data.flatten().view(torch.uint8))

    def read_sflag(self, address: int) -> int:
        return self.sflag.read(address, torch.uint32.itemsize).view(torch.uint32).item()

    def write_sflag(self, address: int, value: int) -> None:
        self.sflag.write(address, torch.tensor([value], dtype=torch.uint32).view(torch.uint8))

    def push_mxu0_weight(self, weight: torch.Tensor) -> None:
        assert weight.shape[0:2] == (self.num_sublanes, self.num_lanes)
        # roll columns to the right
        self.mxu0_weight_buffer = self.mxu0_weight_buffer.roll(self.num_sublanes, dims=1)
        # populate first 7 columns with register contents
        self.mxu0_weight_buffer[:, 0:self.num_sublanes] = weight.reshape(self.num_lanes, self.num_sublanes)

    def push_mxu0_weight_transpose(self, weight: torch.Tensor) -> None:
        assert weight.shape[0:2] == (self.num_sublanes, self.num_lanes)
        # roll columns to the right
        self.mxu0_weight_buffer = self.mxu0_weight_buffer.roll(self.num_sublanes, dims=1)
        # populate first 7 columns with register contents
        self.mxu0_weight_buffer[:, 0:self.num_sublanes] = weight.reshape(self.num_lanes, self.num_sublanes)

    def pop_mxu0_accumulator(self) -> torch.Tensor:
        result = self.mxu0_accumulator[0:self.num_sublanes, :].clone()
        self.mxu0_accumulator = self.mxu0_accumulator.roll(-self.num_sublanes, dims=0)
        return result
