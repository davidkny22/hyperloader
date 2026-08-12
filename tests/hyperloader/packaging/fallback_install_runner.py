"""Exercise an installed universal fallback through positive-worker public execution."""

from hyperloader import DataLoader, _hyperloader


def main() -> None:
    assert _hyperloader.IS_FALLBACK is True
    loader = DataLoader(range(8), batch_size=2, num_workers=2, seed=7)
    try:
        batches = list(loader)
    finally:
        loader.close()
    assert len(batches) == 4
    assert sum(len(batch) for batch in batches) == 8
    print("fallback batches=4 samples=8")


if __name__ == "__main__":
    main()
