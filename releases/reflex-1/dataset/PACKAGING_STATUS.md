# MOBILE_ROUTING9 release status

Private dataset: https://huggingface.co/datasets/cbarkinozer/MOBILE_ROUTING9

All four splits uploaded and downloaded back with matching SHA-256s: train 781, validation 90, calibration 90, test 180 (1,141 total). Exact original train/validation/calibration files recovered from Git; source hashes matched the committed manifest. The release strips review/UI metadata while retaining request/schema/target fields. The 88-row historical development test remains excluded; the 180-case frozen evaluation cohort remains the release test.

All six split pairs have zero overlapping IDs or normalized request text. Semantic/template independence is not guaranteed. Cards and reciprocal model links are updated. See ../hf-dataset-development-upload-20261015.json for revisions and verification hashes. Local upload folder: data/prepared/hf-mobile-routing9/ (ignored by Git).

Both repositories remain private. Author-selected licensing remains pending before public release.
