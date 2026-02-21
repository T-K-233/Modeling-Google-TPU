import torch


class Memory:
    def __init__(
        self,
        name: str,
        size: int,
        base: int = 0x0000_0000,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.base = base
        self.size = size
        self.verbose = verbose
        self.mem: torch.Tensor = torch.zeros(self.size, dtype=torch.uint8)

    def read(self, address: int, size: int) -> torch.Tensor:
        address -= self.base
        assert address + size <= self.size, \
            f"Memory '{self.name}': read out of bounds: {address:04x} + {size} > {self.size:04x}"
        if self.verbose:
            print(f"\033[90m  Memory '{self.name}': read {size} bytes <- 0x{address:04x}\033[0m")
        return self.mem[address:address + size]

    def write(self, address: int, data: torch.Tensor) -> None:
        address -= self.base
        assert data.dtype == torch.uint8, \
            f"Memory '{self.name}': write data must be uint8, got {data.dtype}"
        assert address + data.numel() <= self.size, \
            f"Memory '{self.name}': write out of bounds: {address:04x} + {data.numel()} > {self.size:04x}"
        self.mem[address:address + data.numel()] = data.flatten()
        if self.verbose:
            print(f"\033[90m  Memory '{self.name}': wrote {data.numel()} bytes -> 0x{address:04x}\033[0m")
