# MOBILE_ROUTING9 release status

Uploaded privately: https://huggingface.co/datasets/cbarkinozer/MOBILE_ROUTING9

781 train rows plus 180 frozen test rows, nine routes. Source recovered from origin/main commit 9f28a9b; exact source SHA c01503a65e182b0eb68181fcfcdd0e775c3ea4b9a3039ee9f3abcb4f05d7a2b4. Executed request/schema/target fields preserved; reviewer/UI metadata omitted. Validation, calibration and historical tests excluded; the frozen 180-case test is released separately. Zero ID or normalized-request overlap with the separate frozen 180-case evaluation cohort.

README, manifest, train and test JSONL downloaded at the uploaded revision and SHA-256 verified. Reciprocal dataset/model links added. See ../hf-dataset-test-upload-20261015.json for revisions and hashes. Local upload folder: data/prepared/hf-mobile-routing9/ (ignored by Git).

Both Hub repositories remain private. License selection remains pending before public release. The dataset contains only the phone training component, not the 781-row general mixture ; the 180-case benchmark is now included as the separate test split.

Still needed: exact val.jsonl and calibration.jsonl (90 rows each) from phone-routing-human-reviewed-v2-1080-fixedval-20261013, with executed provenance. The older local calibration file was not substituted.
