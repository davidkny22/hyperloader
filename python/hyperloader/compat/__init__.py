"""Torch-compatible execution modes."""

from typing import Any


def prepare(loader: Any) -> None:
    """Prepare the torch-compatible worker regime selected by the surface."""
    from hyperloader.config import AUTO

    workers = 0 if loader.num_workers is AUTO else int(loader.num_workers)
    if workers == 0:
        from .zero import prepare as prepare_zero

        prepare_zero(loader)
        loader._compat_kind = "zero"
        return
    from .multi import prepare as prepare_multi

    prepare_multi(loader, workers)
    loader._compat_kind = "multi"


def iterate(loader: Any) -> Any:
    """Create an iterator for the prepared compatibility regime."""
    if loader._compat_kind == "zero":
        from .zero import iterate as iterate_zero

        return iterate_zero(loader)
    from .multi import iterate as iterate_multi

    return iterate_multi(loader)


def capture_state(loader: Any) -> dict[str, object]:
    """Capture compatibility state for the prepared worker regime."""
    if loader._compat_kind == "zero":
        from .zero import capture_state as capture_zero

        return capture_zero(loader)
    from .multi import capture_state as capture_multi

    return capture_multi(loader)


def restore_state(loader: Any, state: dict[str, object]) -> None:
    """Restore compatibility state for the prepared worker regime."""
    if loader._compat_kind == "zero":
        from .zero import restore_state as restore_zero

        restore_zero(loader, state)
        return
    from .multi import restore_state as restore_multi

    restore_multi(loader, state)


__all__ = ["capture_state", "iterate", "prepare", "restore_state"]
