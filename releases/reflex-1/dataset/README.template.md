---
language:
- en
pretty_name: MOBILE_ROUTING9
task_categories:
- text-classification
tags:
- synthetic
- human-annotated
- tool-routing
- decision-model
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# MOBILE_ROUTING9

**Local dataset-card template.** Choose the final repository name and license before upload. No license has been assigned by this template.

Planned dataset repository: `cbarkinozer/MOBILE_ROUTING9`. This name covers the phone-routing task; this train-only package does **not** contain the separate 180-case evaluation cohort scored in the model card.

This is the **phone component** of [Reflex-1](https://huggingface.co/cbarkinozer/Reflex-1), a Qwen3.5-0.8B decision-head model. It is not the whole general+phone training mixture and does not contain the benchmark questions.

The source dataset was generated synthetically with `build_phone_routing_dataset.py-v1` and reviewed by one human in Argilla. The reviewer saw proposed labels; this is not independent blind multi-annotator validation. Review metadata was used to quarantine inconsistent or ambiguous cases rather than rewriting the human responses silently. The admitted training component has **781 rows**, approximately balanced over nine routes. Exact per-route counts and source/export SHA-256s are in `manifest.json` after packaging the executed training inputs.

## Fields

- `id`: stable original record identifier.
- `state`: request text.
- `questions`: JSON-encoded decision question/options, compatible with the recorded Unsloth training format.
- `gold`: JSON-encoded route gold label and probabilities.
- `n_questions`: 1.
- `route`: convenience copy of the label.
- `source`: synthetic-human-reviewed-phone-routing.

Routes: `answer`, `get_device_info`, `get_battery_info`, `get_storage_info`, `get_media_info`, `search_local_files`, `search_notes`, `save_note`, `phone_health_check`.

A specific battery, storage or device question uses its specific route; overall condition or multiple live-data categories use phone_health_check. Some records express plausible future phone capabilities. Intent classification accuracy does not establish that a deployed phone tool supports the requested operation. Scenario categories were review metadata, not model prediction targets.

## Splits and use

Only training rows are included. Validation (90), calibration (90), historical test (88 usable cases) and fresh180 test are kept separately and excluded from this release package. Do not move them into training to reproduce the published scores. The historical phone test had already influenced development; fresh180 was reviewed/frozen separately after candidate selection. Template/normalized overlap checks do not prove semantic independence.

```python
from datasets import load_dataset
rows = load_dataset("cbarkinozer/MOBILE_ROUTING9", split="train")
# Pass these rows through the same FastDecisionModel.build_dataset workflow.
```

## Relationship to model training

Reflex-1 trained on 781 phone + 781 general rows (391 typed-decisions, 195 BANKING77, 195 CLINC150), independently from base Qwen3.5-0.8B. Those general data remain separate sources: Apache-2.0 typed-decisions, CC-BY-4.0 BANKING77, CC-BY-3.0 CLINC150. This phone-only dataset does not sublicense those sources. See the model card and pinned mixture recipe for the exact combined training setup.

## Limitations

English synthetic requests, a single reviewer, limited route/schema scope, near-template correlations, and sparse scenario subcategories. Not a general function-argument generator, a safety classifier, or evidence that a full phone assistant works end to end. Any later public benchmark becomes a regression set for future versions.

## License and provenance

Before upload add `license:` to the YAML and include matching LICENSE text after the author selects a license and verifies rights for the generated phone examples. Preserve generator/source revisions, manifest hashes, quarantine policy and the one-reviewer limitations. Remove this template notice and replace the dataset repository placeholder only after checking the packaged files. No reviewer UUIDs, API keys, raw UI backups or personal review notes belong in the dataset repository.
