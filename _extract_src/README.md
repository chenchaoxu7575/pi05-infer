# `_extract_src/` -- NOT-YET-REFACTORED SOURCES

These files are **verbatim copies** of RLinf sources kept here only as the material that
`pi05_infer/engine.py` and `bench/` are extracted from. **They are not part of the package**,
are not importable, and must not be edited in place -- treat them as read-only reference.

| file | origin | rev | lines | md5 |
|---|---|---|--:|---|
| `openpi_action_model.py` | `rlinf/models/embodiment/openpi/openpi_action_model.py` | `cbb9d2fc` | 1824 | `86e98eeb580dc7a74b68db3fab1e3866` |
| `standalone_infer_bench.py` | `benchmarks/pi05_infer/standalone_infer_bench.py` | `cbb9d2fc` | 367 | `2af8d26eab3a6f10dd81c29f5345a2a3` |

Both origins are paths inside the `RLinf-pi05-nsys-profile` tree at rev `cbb9d2fc`.

The md5s are the auditable part: they are what was checked against the copy actually
deployed in the benchmark container that produced the measured numbers, so a reader who
has that tree can confirm these files are the exact material the extraction started from.
`EXTRACTION_NOTES.md` section 0 explains why that check is an md5 rather than a `git diff`.
