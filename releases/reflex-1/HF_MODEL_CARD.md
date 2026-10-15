---
base_model: unsloth/Qwen3.5-0.8B
language:
- en
tags:
- decision-model
- tool-routing
- qwen3.5
- unsloth
- lora
- gguf
datasets:
- LocalLLaMA/typed-decisions
- PolyAI/banking77
- clinc/clinc_oos
---

# Reflex-1

Reflex-1 is a Qwen3.5-0.8B decision model trained to select among supplied options, including nine phone-assistant routes. It uses Unsloth's Clef-style decision head and rank-16 LoRA. It returns structured decisions/probabilities rather than generating an answer or tool arguments. The current Hugging Face repository is private; this is a local release-card draft with completed matched mobile results and release licensing still to be selected by the author.

## Results

### Saved PyTorch adapter

| Benchmark | Cases / decisions | Reflex-1 accuracy | Notes |
| --- | ---: | ---: | --- |
| Fresh phone routing | 180 / 180 | **91.11% (164/180)** | 20 cases per route; macro-F1 90.84% |
| Historical phone routing | 88 / 88 | 96.59% (85/88) | Previously exposed development cohort |
| typed-decisions | 400 / 2,000 | 60.70% | Per-question; exact-record accuracy 10.00% |
| BANKING77 | 462 / 462 | 74.89% | Six cases per class; supervised in-domain |
| CLINC150 | 450 / 450 | 81.11% | Three cases per class; OOS excluded |

These are saved-adapter scores, not phone/GGUF scores. General tests had no reported skipped/truncated decisions in the PyTorch pipeline. These are not a reproduction of Unsloth's published multi-source recipe or Cloudflare's Decision Index.

### Samsung Galaxy A32 / Q4_K_M

On the same frozen 180-case phone cohort, Reflex-1 achieved **91.11% accuracy at 31.36s mean latency**, compared with **54.44% at 35.40s** for the unchanged Qwen app router with prompt caching enabled. This is +36.67 percentage points and 11.39% lower mean latency in these runs. The uncached app router scored 55.56% at 41.14s. These results cover routing under the documented nine-route policy; they do not establish general tool execution quality. Reflex-1 used more memory and showed similar maximum temperatures; no RAM, thermal or energy advantage is established. A shorter schema-format control was faster but much less accurate, so this is not a universal speed claim.

<!-- A32_RESULTS_START -->
Android evaluation covers the selected model and matched cached/uncached app-router comparisons, plus a schema-format control. Beta, historical and general Android runs are deferred. General benchmark scores reported elsewhere were measured with PyTorch on GPU.

| Model | A32 cohort | Cases / decisions | Accuracy | Macro-F1 | Mean latency | Warm p50 / p95 | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 180 / 180 | 91.11% | 91.06% | 31.36 s | 32.34 / 33.87 s | 0 |
| Qwen app router (cache off) | Fresh phone (180) | 180 / 180 | 55.56% | 53.71% | 41.14 s | 41.05 / 44.92 s | 0 |
| Qwen app router (cache on) | Fresh phone (180) | 180 / 180 | 54.44% | 52.52% | 35.40 s | 36.31 / 38.83 s | 0 |
| Qwen schema JSON (cache off) | Fresh phone (180) | 180 / 180 | 23.33% | 22.41% | 24.93 s | 25.44 / 26.70 s | 0 |

| Model | A32 cohort | Battery min–max | AP max | Android thermal status max | Sampled max PSS | RSS high-water |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 28.90–37.50 °C | 52.60 °C | 3.00 | 1090.65 MiB | 1093.00 MiB |
| Qwen app router (cache off) | Fresh phone (180) | 31.60–37.60 °C | 53.10 °C | 3.00 | 796.23 MiB | 838.14 MiB |
| Qwen app router (cache on) | Fresh phone (180) | 29.30–37.50 °C | 52.90 °C | 3.00 | 854.83 MiB | 896.80 MiB |
| Qwen schema JSON (cache off) | Fresh phone (180) | 31.80–37.40 °C | 53.00 °C | 3.00 | 764.54 MiB | 813.60 MiB |

Fresh180 vs qwen-fresh180: accuracy difference +67.78 percentage points; paired bootstrap95% interval [+60.00, +75.00] points (10,000 resamples, seed3407). This interval concerns this reviewed synthetic cohort, not unrestricted real-world requests.

Fresh180 vs qwen-legacy-fresh180: accuracy difference +35.56 percentage points; paired bootstrap95% interval [+27.78, +43.33] points (10,000 resamples, seed3407). This interval concerns this reviewed synthetic cohort, not unrestricted real-world requests.

Fresh180 vs qwen-legacy-cached-fresh180: accuracy difference +36.67 percentage points; paired bootstrap95% interval [+28.89, +44.44] points (10,000 resamples, seed3407). This interval concerns this reviewed synthetic cohort, not unrestricted real-world requests.

Completed runs only; partial counts are not scores. Warm timing excludes first request; phone context2048. See recorded protocol and per-run raw outputs.
<!-- A32_RESULTS_END -->

On the 180-case phone test, GGUF and PyTorch have equal aggregate accuracy but **94.44% route agreement** (10 changed predictions). Mean per-case maximum absolute probability drift is 0.0654; the largest is 0.8273. On this held-out phone cohort, native GGUF ECE is 3.10% (10 equal-width confidence bins), multiclass Brier sum is 0.1338 and hard-label NLL is 0.3178; these describe the measured cohort rather than universal calibration. These differences include quantization and runtime/numerical effects; no test-based calibration or threshold adjustment was made. Aggregate score agreement does not establish probability parity.

The A32 historical 90-case cohort is the original pre-review set; it differs from the 88 usable reviewed cases in the PyTorch historical row. Their scores are not a quantization-parity comparison.

The app baseline preserves its preexisting raw prompt/action grammar and narrower descriptions of currently available tools. The decision model uses the frozen nine-route schema, including some future-capability requests. This compares the documented routing systems; prompt/policy differences and fine-tuning mean the gain cannot be attributed to the decision head alone. Runs were sequential, not randomized repetitions; starting temperatures differed and both reached Android thermal status3.

The earlier **55.6% / 36.8 s** Qwen raw-HTTP result used a historical 90-case set. The fresh 180-case result cannot be compared directly against that percentage. Historical JNI 82% used a different runtime/prompt/cohort and is not a matched baseline. The current evaluation compares actual per-case outcomes and records its prompt/grammar treatment rather than assuming these historical numbers transfer.

## Model and training

- Base: `unsloth/Qwen3.5-0.8B`, independently loaded from base for this run; not continued from reflex-1-beta.
- Data: 781 reviewed synthetic phone requests + 781 general requests (391 typed-decisions, 195 BANKING77, 195 CLINC150). Equal phone/general row counts (typed rows contain five decisions), all general intent classes retained.
- Nine phone routes: answer, get_device_info, get_battery_info, get_storage_info, get_media_info, search_local_files, search_notes, save_note, phone_health_check.
- LoRA rank/alpha 16, base learning rate 2e-4, seed 3407, effective batch 32, max training sequence 4096. See executed run manifests for full settings/head LR rather than assuming defaults.
- Selected training run: [W&B 66wubl55](https://wandb.ai/c-barkinozer/reflexmodels/runs/66wubl55) (tracking was used only for actual training).
- Independent 120-step combined trial; checkpoint 100 selected on validation equal-domain mean, then calibrated on a separate calibration split. Validation phone 98.89%, general 78.39%; those selection scores are not held-out performance.
- Training optimizer time approximately 17.5 min for the selected 120-step run, on RTX 4090 24GB. Setup/validation/calibration/export/billing add time; cost estimates are not final invoices.
- Fresh180 test was reviewed and frozen after candidate selection. Test results did not select checkpoint 100. Single reviewer saw synthetic proposed labels, so human review was not blind independent annotation.

The first unpublished model is now called reflex-1-beta. It was typed-decisions-only; this combined model takes the Reflex-1 name. Do not confuse the two GGUF files: they share the same byte size but have different hashes.

## Files and use

Private repository: [cbarkinozer/Reflex-1](https://huggingface.co/cbarkinozer/Reflex-1).

- `adapter/`: LoRA, `joint_head.safetensors`, joint-head/decision configs, tokenizer and processor files. The head/configs are required; an ordinary LoRA-only loader is insufficient.
- `gguf/model-Q4_K_M.gguf`: 545,161,248 bytes; SHA-256 `abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699`.
- `gguf/mmproj-Q8_0.gguf`: 115,923,136 bytes; SHA-256 `37800be294d8c041e6146516b636414b8a75d0dd0878b08f6e0282e471790f94`. Preserved export sidecar; not needed for these text-only routing requests.

Use a decision-capable Unsloth/llama.cpp runtime and POST to `/v1/systemone`. This is not a `/v1/chat/completions` GGUF. Android tests use the documented build11443 with `--no-repack`; ordinary llama.cpp builds may not expose the Decision API.

A text-only CPU server can use the same flags as the A32 test (adjust the executable/library setup for your platform):

```sh
llama-server -m Reflex-1/gguf/model-Q4_K_M.gguf \
  -c 2048 -t 4 -tb 4 -ngl 0 --fit off -fa off --no-repack \
  --host 127.0.0.1 --port 8091 --jinja
curl http://127.0.0.1:8091/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @decision-request.example.json
```

The release's `decision-request.example.json` contains the full nine-option question from `phone-routing-schema.json`; it contains a generic request, not held-out benchmark text. The expected response shape is `answers.route.choice` and `answers.route.probabilities`, with `usage.output_tokens` equal to0. Runtime source: [the verified llama.cpp revision](https://github.com/ggml-org/llama.cpp/tree/f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a). Benchmark flags are a measured configuration, not a claim of optimal settings for every device.

The Python adapter example needs a compatible Unsloth/PyTorch environment; the recorded reference used an RTX 4090. For Android CPU serving use the verified decision-capable GGUF runtime.

```python
from unsloth import FastDecisionModel
model, tokenizer = FastDecisionModel.from_pretrained(
    "PATH_TO_DOWNLOADED_ADAPTER_FOLDER", load_in_4bit=True,
)
FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(model, tokenizer, "How much free storage do I have?", {
    "route": {
        "type": "choice",
        "instructions": "Select the best route for the request.",
        "criteria": {
            "get_storage_info": "Inspect storage space, usage or capacity.",
            "answer": "Ordinary text response without live phone data.",
        },
    },
})
```

The example illustrates API shape, not the nine-option benchmark schema. To reproduce reported routing scores use the exact committed frozen schema and state strings. Threshold selected option probabilities only after validating calibration for the deployed runtime/domain; do not transfer API confidence thresholds blindly.

## Data provenance and planned dataset link

Phone examples were synthetically constructed and reviewed in Argilla by one human; scenario categories are audit metadata, not learned output classes. The admitted phone training component contains 781 examples; the original 900 train examples include records excluded during review and validation selection. A separate phone training dataset will be linked here after its exact executed inputs, license and final repo name are confirmed. No dataset ID is invented in the YAML above.

General sources:

| Source | Pinned revision | Declared source license |
| --- | --- | --- |
| LocalLLaMA/typed-decisions (all) | `6a7927fa3c8ac114e29f5300eacb86f6f37a6447` | Apache-2.0 |
| PolyAI-LDN/task-specific-datasets, banking_data CSV | `57ec275d8078af65b7731c2a98be812d844a6d6b` | CC-BY-4.0 |
| clinc/clinc_oos (plus, OOS excluded) | `155b9c710419136e17307b80d0a13e68cd46b4ec` | CC-BY-3.0 |

Official training partitions supplied adaptation data; official tests supplied general evaluation. These measure supervised in-domain adaptation, not zero-shot generalization. Exact normalized overlap checks passed, but public benchmark pretraining exposure and semantic template overlap cannot be ruled out. Keep source attribution/licenses intact; the mixture is not uniformly Apache-2.0. Dataset upload instructions are in the local `DATASET_UPLOAD_GUIDE.md`; author must select and document the release license before public upload.

## Limits

Battery and device-info were the weakest fresh PyTorch routes (65% and 75% recall). Indirect/boundary requests were harder than everyday requests. Some schema intents refer to plausible future phone capabilities; correct routing does not mean that the current tool can carry out the request. This model does not generate tool arguments, clarify interactively, execute tools or produce final assistant prose.

The decision-only GGUF does not automatically share a loaded backbone with a chat model. Single-backbone decision/text switching and full Pocket Agents Lab integration remain separate work. Routing latency is not full assistant-turn latency. USB-powered temperature/battery traces do not demonstrate energy savings. Mobile timing depends on runtime, threads, context, request schema, starting temperature and sustained load; see the recorded protocol.

Keep tests out of training/calibration/model selection. If released or used to guide a future model, treat these cohorts as regression benchmarks and collect a new independent final test.
