# Phone routing dataset used for Reflex-1

The folder name retains `phone-training-781` for compatibility; it now contains all original prepared phone splits, preserved byte-for-byte from the verified training backup.

| File | Rows | Role |
| --- | ---: | --- |
| train.jsonl | 781 | Fine-tuning |
| validation.jsonl | 90 | Development and checkpoint selection |
| calibration.jsonl | 90 | Fit confidence calibration |
| test.jsonl | 88 | Previously exposed historical comparison |

The separate fresh 180-case test is already committed in [fresh-phone-test-180](../fresh-phone-test-180/README.md). Do not include validation, calibration or either test in training. Historical test results are not fresh blind evidence; the fresh test has also now been evaluated and becomes regression evidence if used to guide later models.

All four split SHA-256 values and row counts were checked against the original source-manifest.json. Cross-split IDs are disjoint. Per-split hashes/counts are in dataset-manifest.json. template-groups.json preserves automatic grouping provenance, not a proof of semantic independence.

`state` contains the request, `route__label` the route, and serialized `questions`/`gold` preserve the original decision schema/targets. Scenario fields are audit metadata, not trained targets. Synthetic examples were human-reviewed with proposed labels visible. Original exporter caveats/exclusions and its provisional audit status remain intact in source-manifest.json. The selected combined model additionally used 781 general examples; those are not part of this phone corpus.

The separate Hugging Face dataset repository was deleted at the user's request. GitHub is the dataset destination; adapter and GGUF remain in the private model repository.
