"""CUDA registration ownership for retained view-source storage."""

from __future__ import annotations

from typing import Any


class HostRegistration:
    """Register distinct retained source allocations and undo partial failure."""

    def __init__(self, sources: tuple[Any, ...]) -> None:
        self._sources = sources
        self._registered: list[int] = []
        self._registered_bytes = 0

    @property
    def registered_bytes(self) -> int:
        """Return the exact retained capacity registered by this owner."""
        return self._registered_bytes

    def activate(self) -> bool:
        """Register every source allocation or restore the pageable state."""
        import torch

        allocations = _allocations(self._sources)
        if not allocations:
            return False
        runtime = torch.cuda.cudart()
        try:
            for pointer, size in allocations:
                result = runtime.cudaHostRegister(pointer, size, 0)
                if int(result) != 0:
                    self.close()
                    return False
                self._registered.append(pointer)
                self._registered_bytes += size
        except (RuntimeError, TypeError):
            self.close()
            return False
        return True

    def close(self) -> None:
        """Release only registrations acquired by this owner."""
        if not self._registered:
            return
        import torch

        runtime = torch.cuda.cudart()
        for pointer in reversed(self._registered):
            runtime.cudaHostUnregister(pointer)
        self._registered.clear()
        self._registered_bytes = 0


def view_sources(loader: Any) -> tuple[Any, ...]:
    """Return retained source allocations for recognized storage-view mappings."""
    from hyperloader.planner import TensorPlan

    if isinstance(loader._plan, TensorPlan) and not loader._plan.shuffle:
        return (loader.dataset,)
    mapping_id = getattr(loader._plan, "mapping_id", None)
    if mapping_id == "torch-tensor-dataset" and not loader._plan.shuffle:
        return tuple(loader.dataset.tensors)
    if mapping_id == "numpy-memmap" and not loader._plan.shuffle:
        return (loader._execution_dataset._array(),)
    return ()


def _allocations(sources: tuple[Any, ...]) -> tuple[tuple[int, int], ...]:
    allocations: set[tuple[int, int]] = set()
    for source in sources:
        storage = getattr(source, "untyped_storage", None)
        if storage is not None:
            if bool(source.is_pinned()):
                continue
            allocation = storage()
            pointer = int(allocation.data_ptr())
            size = int(allocation.nbytes())
        else:
            interface = getattr(source, "__array_interface__", None)
            if not isinstance(interface, dict):
                continue
            pointer = int(interface["data"][0])
            size = int(source.nbytes)
        if pointer and size:
            allocations.add((pointer, size))
    return tuple(sorted(allocations))
