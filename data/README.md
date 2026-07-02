# Data Policy

This repository does not track SemEval task data, generated result
tables, raw competition zips, or archived submission files. The earlier checkout
stored many of these files as Git LFS pointers, not usable data. Until the
shared-task redistribution terms are explicitly confirmed, keep those files
local-only.

Expected local filenames for camera-ready reruns:

| File | Expected rows | Purpose |
| --- | ---: | --- |
| `dev_track_a.jsonl` | 200 | Track A development triples with labels |
| `test_track_a.jsonl` | 400 | Track A test triples, if labels are released locally |
| `dev_track_b.jsonl` | 479 | Track B development stories |
| `test_track_b.jsonl` | 849 | Track B test stories |
| `synthetic_data_for_classification.jsonl` | 1900 | Synthetic Track A-style training triples |

Previous LFS pointer checksums, retained only to help verify recovered files:

| File | SHA-256 from pointer | Size from pointer |
| --- | --- | ---: |
| `dev_track_a.jsonl` | `d5db7238829291c8c9def924ff3285af61aa0e8c0ec9a2e3a442e09b47dd5568` | 450593 |
| `test_track_a.jsonl` | `27dbd0b9700eb0c4c0801ec6970178fa2c4ff5c1789f0dd2bb2c92d1f77c3313` | 878165 |
| `synthetic_data_for_classification.jsonl` | `4fdcd6e3a7e4bce736308c708b2285230ed479806297f3d3d95c6fca9e989e5b` | 5951293 |

The release checker accepts missing data files but fails if a tracked file is a
Git LFS pointer stub. Use `python scripts/check_release.py --strict-data` when
you have restored the local datasets and want row-count validation.

Local `.jsonl`, `.csv`, and raw zip files under `data/` are gitignored so that
restored task files do not accidentally enter the public release.
