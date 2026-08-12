# Changelog

## Current release

- Provides a Torch-shaped data loader for map-style and stateful iterable sources.
- Preserves coordinate-bound deterministic streams across supported execution tiers.
- Restores map streams across worker-count and divisible world-size changes.
- Supports strict-order and delivery-on-completion contracts.
- Provides native process, declared-thread, in-process, and pure-Python fallback execution.
- Adds adaptive frontier scheduling, persistent cost profiles, controller calibration,
  telemetry, diagnosis, and machine-keeping support.
- Adds Torch compatibility mode, distributed rank placement, gloo validation, and
  free-threaded CPython routing.
- Adds image, text, NumPy, Arrow, Parquet, tensor-view, pinned-delivery, and stage pipelines.
- Defines native platform wheels and a deterministic universal fallback wheel.
