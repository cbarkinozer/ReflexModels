# Dataset registry

Only datasets with verified licenses may be added to training or evaluation.
Do not commit private or identifiable support text.

| Name | URL | License | Languages | Intended use | Turkish provenance | Version / notes |
|---|---|---|---|---|---|---|
| MASSIVE 1.0 | https://github.com/alexa/massive | CC BY 4.0 (data) | en-US, tr-TR, 49+ others | fixed-label intent baseline train/validation/test | translated/localized parallel | Prepared locally on 2026-09-21; 33,042 aligned-locale records; 278 normalized-text duplicates across official splits flagged. |
| XNLI 1.0 | https://github.com/facebookresearch/XNLI | CC BY-NC 4.0 | en, tr, 13 others | non-commercial NLI evaluation only | translated parallel | 2,500 dev / 5,000 test per language; cannot support commercial releases. |
| ReflexBench-TR negation seed | local | CC0-1.0 | tr | frozen evaluation seed | native | Hand-authored minimal pairs; never train on it. |

When registering a dataset, record a direct source URL, exact license/version,
and any commercial-use restriction before use. Never combine the XNLI
evaluation data into a commercially releasable training mixture.
