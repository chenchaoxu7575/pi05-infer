# `_extract_src/` — NOT-YET-REFACTORED SOURCES

These files are **verbatim copies** of RLinf sources kept here only as the material that
`pi05_infer/engine.py` and `bench/` are extracted from. **They are not part of the package**,
are not importable, and must not be edited in place — treat them as read-only reference.

| file | origin | rev | lines | md5 |
|---|---|---|--:|---|
| `openpi_action_model.py` | `RLinf-pi05-nsys-profile/rlinf/models/embodiment/openpi/openpi_action_model.py` | `cbb9d2fc` | 1824 | `86e98eeb580dc7a74b68db3fab1e3866` |
| `standalone_infer_bench.py` | `RLinf-pi05-nsys-profile/benchmarks/pi05_infer/standalone_infer_bench.py` | `cbb9d2fc` | 367 | `2af8d26eab3a6f10dd81c29f5345a2a3` |

Verified byte-identical to the copy actually deployed in the `pi05bench` container at
`/workspace/rlinf_pub/RLinf-pi05-nsys-profile` (i.e. the tree that produced the measured
numbers recorded in the Stage 0 commit).
