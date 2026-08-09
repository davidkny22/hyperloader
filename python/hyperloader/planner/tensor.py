"""Planning contract for contiguous pre-tokenized tensors."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloader import _hyperloader


@dataclass(frozen=True, slots=True)
class TensorPlan:
    """Deliver storage-contiguous tensor batches as views."""

    length: int
    shuffle: bool

    def index(self, root_seed: int, epoch: int, position: int) -> int:
        """Map one sampler position to its dataset index."""
        if not 0 <= position < self.length:
            raise IndexError("sampler position is outside the dataset")
        if not self.shuffle:
            return position
        return _hyperloader._permutation_index(root_seed, epoch, self.length, position)


def build_tensor_plan(dataset: object, shuffle: bool | None) -> TensorPlan:
    """Build a view-capable plan for a recognized tensor dataset."""
    length = len(dataset)  # type: ignore[arg-type]
    return TensorPlan(length=length, shuffle=bool(shuffle))
