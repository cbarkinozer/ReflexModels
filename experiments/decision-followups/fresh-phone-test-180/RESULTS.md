# Fresh phone test results - 2026-10-15

Fixed reflex-1 combined checkpoint 100 evaluated on the previously frozen 180-case test. Source evaluator commit `570b4b26c8f834902e05b960a010f505dbfda3c6`; launcher commit `b918556`. Test SHA-256 `2d18133e44a9f9db82fe80099b642f00cd6a12ebf643dc043d1f89f9ddfa209b`. No training, calibration fitting or checkpoint selection occurred on this test.

**164/180 correct = 91.11% accuracy; macro-F1 = 90.84%.** Wilson 95% accuracy interval: 86.05%-94.45%, assuming independent examples from this test distribution. Synthetic coverage and shared drafting patterns limit that assumption and population generalization. Native batch accuracy agrees with per-request serving accuracy. No skipped or truncated decisions. Native ECE 0.05524; loss 0.33116.

| Route | Correct / 20 | Recall | F1 |
| --- | ---: | ---: | ---: |
| answer | 18 | 90.00% | 87.80% |
| get_battery_info | 13 | 65.00% | 78.79% |
| get_device_info | 15 | 75.00% | 81.08% |
| get_media_info | 20 | 100.00% | 95.24% |
| get_storage_info | 19 | 95.00% | 95.00% |
| phone_health_check | 20 | 100.00% | 86.96% |
| save_note | 20 | 100.00% | 95.24% |
| search_local_files | 19 | 95.00% | 97.44% |
| search_notes | 20 | 100.00% | 100.00% |

Coverage: everyday 88/90 (97.78%); indirect 38/45 (84.44%); boundary 38/45 (84.44%). The main weakness is battery routing (13/20), including five battery requests sent to phone_health_check; hardware specs also underperform (15/20). Four routes have perfect recall on only 20 examples each; this does not establish perfect precision or reliability. Negation and competing domain vocabulary caused mistakes such as a mnemonic request being interpreted as saving a note, and an audio call's charge estimate being interpreted as media information.

Every request, probability vector and predicted label is saved in `results/predictions.jsonl`; all 16 errors in `results/errors.json`; metrics/confusion matrix in `results/summary.json`. These are research routing decisions, not a full tool-execution benchmark. No unmodified prompted 0.8B tool-calling baseline was measured on this cohort, so it does not prove superiority to that baseline. GGUF parity and on-phone testing remain pending.

Authorship/review limitations: synthetic author previously saw project examples, human labels were validated with author suggestions, no labels were changed, and zero exact overlap does not prove semantic/pretraining independence. If results inform future improvements, this cohort is a regression benchmark for the next model; reserve a new sealed set for that model's final claims.

Provider: Vast instance 55996354, RTX 4090, host 34031, machine 136736, SSH ssh6.vast.ai:10788. Snapshot 2026-10-14 21:57:08 UTC (2026-10-15 local): duration 687.80 s, total rate $0.47703531/hour (base $0.43999827 + storage $0.03703704), elapsed-rate estimate $0.09114. Network tariffs/counters are saved without inferring final bill units. Final invoice/shutdown time remains unknown; pod is left running. Model loading plus batch/serving evaluation took 88.25 seconds; rental includes setup and transfer. No W&B runs created.

Backup: full adapter/head/tokenizer/configs, frozen test/provenance, evaluator, environment, predictions/results, logs and sanitized GPU/cost inventory are archived remotely as `fresh180-closeout-20261015.tar.gz`, 86345715 bytes, SHA-256 `68b0400f8047989c95b8919ee5c439bb85352a7c876b6d71824f97edd77d4e26`. Local destination: `pocket-agents-lab/runs/decision-followups-vast-20261013/fresh180-20261015/`. Local archive hash and all 28 inventoried files verified successfully. Prediction file hash also matches the remote evaluator summary. The pod is disposable for this completed evaluation; no stop/destroy action was taken. Credentials, virtualenv and reproducible caches excluded; original full training/base-cache backups remain local.
