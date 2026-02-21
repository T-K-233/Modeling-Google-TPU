import torch

from memory import Memory


class ArchState:
    def __init__(
        self,
    ) -> None:

        self.pc = 0
        self.next_pc = 0
        self.hbm_size = 1024 * (1 << 20)  # 1024 MiB
        self.vmem_size = 128 * (1 << 20)  # 128 MiB
        self.smem_size = 1 * (1 << 20)  # 1 MiB
        self.sflag_size = 256  # 256 bytes (64 words)
        self.num_x_registers = 64
        self.num_v_registers = 64
        self.v_register_shape = (8, 128)

        self.initialize_buffers()

    def initialize_buffers(self) -> None:
        self.hbm = Memory("HBM", self.hbm_size)
        self.vmem = Memory("VMEM", self.vmem_size)
        self.smem = Memory("SMEM", self.smem_size)
        self.sflag = Memory("SFLAG", self.sflag_size)

        self.xreg: dict[str, int] = {}
        for index in range(self.num_x_registers):
            self.xreg[f"s{index}"] = index

        self.vreg: dict[str, torch.Tensor] = {}
        vreg_size = self.v_register_shape[0] * self.v_register_shape[1] * torch.float32.itemsize
        for index in range(self.num_v_registers):
            self.vreg[f"v{index}"] = torch.zeros(
                vreg_size,
                dtype=torch.uint8,
            )

    def read_xreg(self, src: str) -> int:
        return self.xreg[src]

    def write_xreg(self, dest: str, value: int) -> None:
        self.xreg[dest] = value

    def read_vreg(self, src: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (
            self.vreg[src]
            .view(dtype)
            .reshape(*self.v_register_shape)
        )

    def write_vreg(self, dest: str, value: torch.Tensor) -> None:
        assert (
            value.numel() == self.v_register_shape[0] * self.v_register_shape[1]
        )
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
