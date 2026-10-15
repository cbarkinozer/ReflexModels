# Exact 781-row phone training split

`train.jsonl` is the unmodified phone training file used for the phone-only and selected combined Reflex-1 fine-tunes. 781 rows, nine routes. SHA-256: `c01503a65e182b0eb68181fcfcdd0e775c3ea4b9a3039ee9f3abcb4f05d7a2b4`. Validation, calibration and test examples are not included. The combined run also used 781 general examples, which are not included here.

`state` contains the request, `route__label` the route, and serialized `questions`/`gold` preserve the original decision schema/targets. Scenario fields are metadata, not trained scenario targets. Synthetic examples were human-reviewed with proposed labels visible. Original exporter caveats/exclusions and split hashes are preserved in source-manifest.json; its provisional status is retained as original provenance. Dataset-manifest.json records route counts.

The accidentally created separate Hugging Face dataset repository was deleted at the user's request. This GitHub directory is the dataset destination. Model adapter and GGUF remain in the private model repository.
