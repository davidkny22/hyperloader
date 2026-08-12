# hyperloader

hyperloader is a PyTorch data loader with exact deterministic streams, exact coordinate
resume, persistent execution, and adaptive scheduling. Existing map-style datasets work
through a `DataLoader` surface shaped like `torch.utils.data.DataLoader`. Stateful iterable
sources can opt into exact replay and resume.

The default path isolates arbitrary dataset code in persistent worker processes. Faster
thread and native paths are entered only through an explicit declaration or a recognized
structure. Execution placement, worker count, scheduling order, and controller decisions do
not change results inside the documented determinism boundary.

## Install

Install the wheel matching the active Python and platform:

```console
python -m pip install hyperloader-0.1.0-<python>-<platform>.whl
```

When no native wheel matches, install the universal fallback artifact:

```console
python -m pip install hyperloader-0.1.0-py3-none-any.whl
```

PyTorch is the runtime dependency. Native wheels are defined for CPython 3.10 through 3.15,
including 3.14t and 3.15t, on Linux, macOS, and Windows. The universal fallback carries the
same public contracts through a pure-Python process engine.

## Use

```python
from hyperloader import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    seed=1701,
)

for batch in loader:
    train(batch)

loader.close()
```

Checkpoint a map-style stream through the loader itself:

```python
iterator = iter(loader)
batch = next(iterator)
state = loader.state_dict()

restored = DataLoader(dataset, batch_size=64, num_workers=4, seed=1701)
restored.load_state_dict(state)
```

Changing worker count or distributed world size preserves map-style continuation when the
recorded global batch divides into the new topology. `set_epoch(epoch)` is the explicit
replay route. A new iteration after abandoning an iterator that delivered a batch advances
the epoch automatically.

Declare `thread_safe=True` only when dataset code neither mutates shared state without
synchronization, depends on process isolation, nor draws from unprovided global RNG sources.
Use `hyperloader.rng("random")`, `hyperloader.rng("numpy")`, or
`hyperloader.rng("torch")` for coordinate-bound draws.

## Contracts

- Strict-order delivery is deterministic across execution tiers and controller schedules.
- Delivery on completion preserves batch content and composition, but delivery sequence is
  execution-dependent.
- Torch compatibility mode reproduces the pinned Torch behavior for its verified platform
  and minor-version matrix.
- Iterable exact resume requires the stateful-source obligations documented by the public
  protocol. Iterable topology elasticity is not claimed.
- Code that observes worker identity, process identity, wall clocks, unprovided RNG sources,
  or hash-order-dependent collections is outside the native determinism boundary.
- Holding a delivered batch cannot corrupt later data. Memory growth remains visible through
  diagnosis and telemetry.

## Diagnose and verify

`hyperloader.diagnose(loader)` returns a machine-readable report covering realized blocking,
frontier saturation, worker resources, controller decisions, ceiling binds, GIL restoration,
and promotion evidence. `hyperloader.verify(...)` checks public contract behavior for a
dataset configuration.

Telemetry is passive by default. Active diagnosis probes are opt-in and report their own
cost separately.

## Measured results

The following DGX Spark results are verified registry claims. Each cell used the installed
public import, equal tuning, alternating paired order, warmup outside timing, and a
deterministic bootstrap 95 percent interval.

| Comparison | Mean result | 95% interval |
|---|---:|---:|
| Fixed-text identity throughput over Torch | 5.488% faster | [5.376%, 5.604%] |
| NumPy identity throughput over Torch | 18.339% faster | [18.219%, 18.464%] |
| Arrow identity throughput over Torch | 18.601% faster | [18.472%, 18.728%] |
| Fixed-text compute penalty against a free resident feeder | 0.130% | [0.015%, 0.314%] |
| Fixed-text bandwidth penalty against a free resident feeder | 0.027% | [-0.058%, 0.085%] |

Identity comparisons use four workers and batch shape `int64[64,512]`. The resident feeder
uses at least eight times the 24 MiB last-level cache. The overhead cells run one uninterrupted
GPU workload for 90 seconds and swap feeders at midpoint. The compute result contains 19
pairs; the bandwidth result contains 10 pairs.

These figures describe the named Spark configuration. They do not claim the same magnitude
on another machine or workload. Per-machine calibration selects execution using locally
measured costs.

## Platform status

Windows x86_64 and Linux aarch64 native execution have hardware-backed assurance. Linux
aarch64 also covers CPython 3.14t and 3.15t routing, stress, and gloo execution. The macOS
implementation and workflow exist, but macOS execution remains unverified until a runner is
available. No macOS result is used by a published performance claim.

## Prior art and credits

hyperloader builds on published and shipped ideas including Philox counter RNGs, format-
preserving permutation, Torch process transport and loader semantics, SPDL staged threading,
FFCV arenas, LPT scheduling, StatefulDataLoader resume, MosaicML canonical partitioning,
MinatoLoader head-of-line analysis, and analytical pipeline controllers such as Plumber.
The precise adopted mechanisms and differences are recorded in [NOTICE](NOTICE).

## License

hyperloader is available under the [MIT License](LICENSE).
