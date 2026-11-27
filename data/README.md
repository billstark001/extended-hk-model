# Data

`fixtures/` is reserved for small versioned test inputs. `manifests/` records external datasets and legacy run locations. Generated research data belongs in `outputs/` and is not versioned.

The 2026-11-27 EHK observation migration has its durable local copy under
`E:/ResearchData/ehk/recommendation/20261127/`. Each of `staged`, `staged_timing`
and `recommender_variants` contains the canonical `analysis.sqlite`, frozen plan
and executable binding. The raw histories remain in their original
`F:/ehk_cache/run/<profile>` directories.

`imported-json.zip` and `json-archive-receipt.json` preserve only the authenticated
task JSON and per-arm observation JSON already imported into SQLite. They do not
contain raw trajectories, event databases, snapshots or detailed exposure traces.
Receipts record per-file SHA-256 verification and whether source JSON was removed.
Normal analysis reads SQLite; ZIP files are explicit audit/restore artifacts.
These local databases and archives stay outside Git.
