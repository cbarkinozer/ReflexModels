---
language:
- en
pretty_name: MOBILE_ROUTING9
task_categories:
- text-classification
task_ids:
- intent-classification
size_categories:
- n<1K
tags:
- mobile
- intent-classification
- tool-routing
- synthetic
- human-annotated
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: test
    path: data/test.jsonl
---

# MOBILE_ROUTING9

MOBILE_ROUTING9 is an **English text classification dataset for mobile-assistant request routing**. Each request is assigned one of nine intent labels: answering directly, inspecting device information, inspecting battery or storage, checking media, finding files or notes, saving a note, or checking overall phone condition.

The current release contains **961 synthetic, human-reviewed examples: 781 training and 180 test examples**. It supports training and evaluating intent classifiers, including conventional text classifiers and models that score a supplied set of routing options.

## Task and labels

Given an English request, predict its single `route` label. Route names describe the intended action; they do not guarantee that a particular phone or application implements it.

| Label | Intended request |
| --- | --- |
| `answer` | General knowledge, arithmetic, writing or conversation; also requests requiring clarification when no stable tool intent is identifiable. |
| `get_device_info` | Device identity, operating system, hardware specifications, capabilities, ratings or certifications. |
| `get_battery_info` | Battery level, charging, temperature, health, consumption or runtime estimates. |
| `get_storage_info` | Storage space, capacity, usage, or internal/external storage status. |
| `get_media_info` | Current or recent media playback, metadata, availability or playback settings. |
| `search_local_files` | Find device files by name, type, content, time, location or provenance. |
| `search_notes` | Retrieve a note or information previously saved in the notes system. |
| `save_note` | Store new information following an explicit request to save or remember it. |
| `phone_health_check` | Overall phone condition or readiness, or a request combining two or more live phone-data categories. |

**Label boundaries:** a battery-only, storage-only or device-only request takes its specific route. Overall condition and requests spanning multiple live-data categories take `phone_health_check`. Retrieving existing information uses `search_notes`; explicitly storing new information uses `save_note`. Clarification is represented by `answer`, not a separate tenth class.

Illustrative examples below explain the policy; they are not additional dataset rows:

| Request | Label |
| --- | --- |
| “What percentage of my battery remains?” | `get_battery_info` |
| “Find the note where I saved my packing list.” | `search_notes` |
| “Save a note: pick up the parcel tomorrow.” | `save_note` |
| “Check both my battery and available storage.” | `phone_health_check` |

## Splits and class distribution

| Label | Train | Test |
| --- | ---: | ---: |
| `answer` | 89 | 20 |
| `get_battery_info` | 82 | 20 |
| `get_device_info` | 89 | 20 |
| `get_media_info` | 87 | 20 |
| `get_storage_info` | 88 | 20 |
| `phone_health_check` | 86 | 20 |
| `save_note` | 86 | 20 |
| `search_local_files` | 86 | 20 |
| `search_notes` | 88 | 20 |
| **Total** | **781** | **180** |

Training is approximately balanced; test is exactly balanced. Separate validation and calibration sets of 90 examples each exist in the original experiment but are **not included in this release yet**. An older test cohort used during development is excluded.

The test split was collected and frozen separately after selection of the initial model checkpoint. No ID or normalized-request overlap was found between the released train and test splits. This check does not establish semantic or template independence.

## Loading and basic use

```python
from datasets import load_dataset

dataset = load_dataset("cbarkinozer/MOBILE_ROUTING9")
train_texts = dataset["train"]["state"]
train_labels = dataset["train"]["route"]
test_texts = dataset["test"]["state"]
test_labels = dataset["test"]["route"]
```

Use `state` as the input text and `route` as the classification target. Labels are strings, so encode them if your classifier requires integer class IDs. While the repository is private, authenticate with an account that has access.

The additional `questions` and `gold` fields preserve a structured decision-model representation. They are optional for ordinary text classification:

```python
import json

row = dataset["train"][0]
questions = json.loads(row["questions"])
gold = json.loads(row["gold"])
assert gold["route"]["label"] == row["route"]
```

## Data fields

| Field | Description |
| --- | --- |
| `id` | Stable source example identifier. |
| `state` | English request text; the classification input. |
| `route` | One of the nine intent labels; the classification target. |
| `questions` | JSON-encoded route instructions and descriptions of the nine candidate labels. |
| `gold` | JSON-encoded reference label and one-hot target probabilities. These are annotation targets, not measured human certainty or model confidence. |
| `n_questions` | Always 1: one routing question per example. |
| `source` | Synthetic, human-reviewed phone-routing provenance label. |

Do not expose `route`, `gold` or label-bearing IDs to a classifier as input features. The shared question schema can be supplied to systems that classify against described options.

## Collection and annotation

Requests were generated synthetically for a phone-assistant routing policy and reviewed by one human using Argilla. The training pool was filtered during review and split preparation, leaving the 781 admitted training examples. The separate test cohort contains everyday and indirect/boundary requests across the same nine routes.

The reviewer saw proposed labels. This was **not blind independent annotation**, and inter-annotator agreement was not measured. Labels express the author's intended routing policy rather than a universal ground truth about assistant behavior. Scenario categories used during review are audit metadata, not extra target classes.

The release preserves the admitted request text, decision schemas, targets and identifiers while excluding reviewer/UI metadata. File hashes, source revision and export details are recorded in [manifest.json](manifest.json).

## Evaluation recommendations

Report **accuracy and macro-F1**, together with per-class recall or a confusion matrix to show boundary errors. State whether a system uses only request text or also the supplied label descriptions. Keep preprocessing and label mappings consistent across models.

Use the test split for final evaluation, not training, calibration, threshold fitting or checkpoint selection. If test results guide subsequent development, treat it as a regression benchmark and collect additional unseen examples for fresh generalization claims. Published scores on this small balanced synthetic cohort do not establish performance on naturally distributed phone traffic.

## Intended use and limitations

Intended for research and development of English mobile-assistant intent classification and tool selection. It does not evaluate tool arguments, actual tool execution, permission handling, final answer quality or a complete assistant turn.

Coverage is limited to nine routes, synthetic phrasing and a single reviewer's policy. Some requests assume plausible future phone capabilities. Similar templates may appear across examples despite exact-overlap checks. There is no dedicated out-of-scope class: `answer` combines direct responses and requests needing clarification. Results may differ for real users, additional languages, new tools or a different routing policy.

## License and related work

A dataset redistribution license has not yet been selected. License metadata will be added before public release.

[Reflex-1](https://huggingface.co/cbarkinozer/Reflex-1) is one model trained using this dataset's phone training split. Its separate general training sources are not included in MOBILE_ROUTING9. See that model's card for its experiments and results.
