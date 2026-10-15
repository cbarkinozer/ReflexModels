# Continue on desktop - 2026-10-15

User will perform ADB testing on the desktop. The laptop task was to preserve and push records; do not restart annotation or training. Pull GitHub main, then follow [TODO.md](TODO.md). The model package is now uploaded privately to Hugging Face; public publication remains pending.

## Model and measurements

Current model is **reflex-1**, combined phone + general Qwen3.5-0.8B decision model, selected checkpoint 100 from the independent 120-step run. Original single-domain release is reflex-1-beta. This model predicts nine phone routes; scenario categories were review metadata, not training targets.

Fresh human-reviewed test: 180 cases / 20 per route. PyTorch reference: **91.11% accuracy (164/180), 90.84% macro-F1**. Battery recall 65%, device-info recall 75%; everyday 97.78%, indirect/boundary 84.44% each. No skipped/truncated cases. See [fresh results](fresh-phone-test-180/RESULTS.md), including limitations. The test is synthetic and human review used proposed labels. Do not claim superiority over unmodified 0.8B tool calling; that baseline was not measured on this cohort.

## Files delivered through Git

- `experiments/decision-followups/fresh-phone-test-180/test.jsonl`: frozen 180 requests and gold labels. SHA-256 `2d18133e44a9f9db82fe80099b642f00cd6a12ebf643dc043d1f89f9ddfa209b`.
- `fresh-phone-test-180/schema.json`: exact training/evaluation route schema. Use it unchanged.
- `fresh-phone-test-180/results/predictions.jsonl`: PyTorch serving reference probabilities/labels per case.
- `fresh-phone-test-180/results/summary.json`, `errors.json`: scores, confusion matrix and 16 failures.
- `fresh-phone-test-180/results/records/`: full evaluation/setup logs, environment, GPU/cost and backup inventory.
- `releases/reflex-1/README.md`, `gguf-manifest.json`: model provenance and GGUF hash inventory.

## Binaries to locate or transfer separately

Git pull does **not** download weights. On the laptop, GGUF files are in:

`C:/Users/barkin/Documents/GitHub/ReflexModels/experiments/decision-followups/gguf-combined-step100/`

- `model-Q4_K_M.gguf`: 545161248 bytes, SHA-256 `abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699`.
- `mmproj-Q8_0.gguf`: 115923136 bytes, SHA-256 `37800be294d8c041e6146516b636414b8a75d0dd0878b08f6e0282e471790f94`.

Selected adapter/head/tokenizer/calibration are in the verified archives under laptop `C:/Users/barkin/Documents/GitHub/pocket-agents-lab/runs/decision-followups-vast-20261013/`. Main original archive: `reflex-decision-artifacts-20261013.tar.gz`; selected adapter member: `reflex-decision/runs/combined-balanced-scratch-120step-s3407/adapter/`. The fresh evaluation backup is `fresh180-20261015/fresh180-closeout-20261015.tar.gz` (86345715 bytes; SHA-256 `68b0400f8047989c95b8919ee5c439bb85352a7c876b6d71824f97edd77d4e26`). All 28 inventory files verified. Original base-cache archive remains alongside the full training archive.

## Desktop benchmark and next release

User connects A32 through ADB. Use Clef-compatible llama.cpp/Decision API runtime; check the existing reflex-1-beta invocation for `/v1/systemone` and required `--no-repack`. This model is a decision-head export, not a chat-completions model. Benchmark the same frozen 180 cases/schema; save every response and compare with PyTorch. Keep score differences visible rather than assuming quantization parity. Measure cold/warm latency, p50/p95, memory and sustained heat/battery behaviour, with model/runtime hashes and settings. Then commit results.

If successful, prepare Hugging Face adapter **with decision head and required configs/calibration/tokenizer**, GGUF and dataset cards; document licenses, provenance, splits, real benchmark results and limitations. Choose the dataset name later. User will authorize publication after review. Keep test out of training/calibration/model selection. Once public or used to guide improvements, it is a regression benchmark for later versions.

Latest GPU was instance 55996354, SSH ssh6.vast.ai:10788, rate about $0.477/hour. Backup complete; user may close it. Agent did not stop/destroy it.

## Download from private Hugging Face instead of copying laptop files

The transfer package is now at https://huggingface.co/cbarkinozer/Reflex-1 (private), revision `157a0d6ed5a218352bcd5b6bed57dbef090b151d`. Authenticate on desktop, then:

```powershell
hf auth login
hf download cbarkinozer/Reflex-1 --revision 157a0d6ed5a218352bcd5b6bed57dbef090b151d --local-dir Reflex-1
```

Use `Reflex-1/gguf/model-Q4_K_M.gguf` and `Reflex-1/gguf/mmproj-Q8_0.gguf` for the A32 benchmark. `adapter/` contains the selected adapter/head/configs/calibration/tokenizer. Merged standalone PyTorch weights are absent; the adapter requires the Qwen base model. Dataset/test files remain on GitHub. Remote weights were verified by size and LFS SHA-256 after upload.


The exact 781-row phone training split is now committed to GitHub in `experiments/decision-followups/phone-training-781/train.jsonl`. The mistakenly created separate Hugging Face dataset repository has been deleted. Model adapter and GGUF remain in the private model repository.
