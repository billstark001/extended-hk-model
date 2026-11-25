# Research-sized benchmark corpus

This module does not change production storage or original data. It opens the local analysis SQLite source in read-only mode and reads saved summary exports and NPZ grids. Dataset preparation and byte-size estimates must run outside database timing windows. Missing sources cause an explicit error.

The corpus has three profiles:

| Profile | Documents | Data status |
| --- | ---: | --- |
| `actual_available` | 496 | Real completed staged-summary exports, transposed into array-bearing documents; numerical values are retained. |
| `full_paper` | 14,099 | Calibrated synthetic extension using every original-paper task coordinate: 1,600 epsilon + 12,288 gradation + 11 mechanism + 200 replicate. |
| `versioned_2x` | 28,198 | The same full-paper coordinates with two distinct synthetic analysis revisions. |

The corpus does not include the original main-scan wide database or bulk raw runs. Consequently, `full_paper` is **not the recovered full scientific dataset**. No generated trajectory should be used as a model result or model validation.

## Measured calibration

- The local final analysis smoke database contains three real 16-node/16-step statistics documents. They have 61 fields and encoded sizes 10,866–11,045 bytes; metadata is 960–964 bytes, protocol 3,480 bytes, and raw-file provenance 1,886–1,892 bytes. These documents define schema/provenance overhead, not the distribution of full-run trajectory lengths.
- The 496 real staged reference summaries each contain 129 observations. Their 3,824 real follow-up arms each contain 33 observations. All 4,320 real arm summaries have 500 final opinions.
- Eleven saved mechanism NPZ files each contain 27 numeric arrays: three vectors of length 11, one length-201 grid, and 23 matrices of shape `(11,201)`, totaling 408,696 numeric bytes per original file.

`full_paper` transfers the observed 33/129-point mixture (3,824/496) to the original-paper schema. This transfer is an explicit storage-size assumption; it is not a measurement of adaptive sampling in the offline main scan. Only the 11 mechanism documents receive the measured probe-grid shapes. Float64 curves, clustered final opinions, source-layout identities and analysis revisions vary deterministically per document. Independent small perturbations avoid a corpus made by repeating one compressible document. Synthetic horizons between 1,024 and the original 20,000-step cap are not inferred stopping-time estimates.

`actual_available` retains the real `separation`, concordance and exposure counter names; these are not relabeled as the main paper's `I_p/I_h`. It contains the real reference and all available stage/arm histories from each saved result. It does not claim that the separately stored raw simulation files were loaded.

## Shared interface

```python
from experiments.storage_benchmark.corpus import (
    get_profile, iter_documents, document_at,
)

description = get_profile("full_paper")
first = document_at(0, "full_paper")
for document in iter_documents("full_paper", limit=32):
    pass
```

All adapters receive the same `_id`, top-level query scalars, `metadata`, `protocol`, `source`, and a flat `payload` dictionary containing only NumPy arrays. Named paths retain observation meanings, for example `reference.pathway.separation` and `stages.early.hold.exposure.rewires`. Scalars use the production names `origin`, `mode`, `analysis_id`, `recsys_type`, `decay`, `rewiring`, `repost_rate`, `tolerance`, `post_retain_count`, `complete`, `stale`, and `name`. Arrays must pass through one common recursive codec and digest policy for every adapter; the corpus itself does not select an encoding or compressor.

`get_profile()` includes measured source paths and SHA-256 hashes, real schema row hashes, measured dimensions/counts, the fixed generation seed, and every synthetic assumption. First-revision documents are identical across `full_paper` and `versioned_2x`; a second revision keeps the task identity but changes its analysis identity and numerical payload. Generation is independent of access order and worker scheduling.

The current generation revision is `research-storage-corpus-v2`. Mechanism array names are traversed in sorted order before consuming their per-document random stream. Therefore a source pack written with sorted JSON keys and reloaded by a worker produces the same array dtype, shape and bytes as direct generation, including the second analysis revision. Version 1 lacked this guarantee for mechanism grids. Existing version-1 benchmark outputs are retained as historical outputs; a version-2 campaign must prepare a fresh source pack and catalog instead of reusing their expected digests.

All 496 transformed actual documents were inspected: their arrays occupy 62.92 MB before encoding, and their JSON with the existing NumPy-NPY/zlib/base64 codec occupies 52.12 MB. They have 248–279 array fields per document. A stratified synthetic sample (48 documents per large task family and all 11/22 mechanism documents) suggested approximately 256 MB for `full_paper` and 524 MB for `versioned_2x` using that same codec. The normal 33-point documents are about 17 KB, 129-point documents about 28 KB, and mechanism documents about 540–553 KB. Synthetic totals are preparation-time extrapolations, not database measurements; they vary by codec and sampling and do not include backend indexes, journals, page overhead or peak transient memory. They include the sparse mechanism-grid tail with its actual task-family weight.
