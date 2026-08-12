# Contributing

Contributions are welcome when they preserve hyperloader's observable contracts and include
evidence proportionate to the change.

## Before opening a change

Open an issue before changing the public API, a deterministic contract, a serialized state
shape, a planner mapping, or a published performance claim. Correctness fixes, focused
performance improvements, platform fixes, tests, and documentation corrections can proceed
directly.

Keep changes narrow. Product source files contain product code only, and tests mirror the
package or engine module they exercise. Preserve stable public re-exports when moving an
internal implementation.

## Verification

Run the focused tests first, then the affected installed public-path gate. Python changes
must pass Ruff. Rust changes must pass Rustfmt, Clippy, and the affected native tests. A gate
change includes measured coverage, a named negative, and a planted fault that demonstrably
turns the verifier red.

Performance changes must state the workload, reference, versions, batch and worker tuning,
warmup, timing window, order control, machine state, commit, and uncertainty interval. Both
sides receive equal tuning. Preserve raw measurements with the report.

## Commits

Use conventional commits with a scope, such as `fix(arena): retain held mappings`. Every
commit has a substantive body that explains what changed and why. Do not add tool or AI
trailers to commit messages.

## Attribution

Preserve [NOTICE](NOTICE). A contribution that adopts a mechanism from another project or
paper names the source, identifies what was taken, and explains what differs. Confirm the
license of every included file and dependency.

## Security and correctness reports

Reduce reports to the smallest dataset, sample coordinate, execution tier, and configuration
that reproduces the behavior. Include the operating system, Python and Torch versions,
expected result, actual result, and a complete command. Never include credentials or private
training data.
