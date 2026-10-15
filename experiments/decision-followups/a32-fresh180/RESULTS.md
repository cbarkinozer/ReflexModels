# Samsung A32 benchmark results

Only completed jobs have scores; pending jobs are explicit.

| Model | A32 cohort | Cases / decisions | Accuracy | Macro-F1 | Warm p50 / p95 | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone180 | 180 / 180 | 91.11% | 91.06% | 32.34 / 33.87 s | 0 |
| Qwen raw JSON | Fresh phone180 | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | Fresh phone180 | 0 completed | pending | pending | pending | pending |
| reflex-1 | Historical phone90 | 0 completed | pending | pending | pending | pending |
| Qwen raw JSON | Historical phone90 | 0 completed | pending | pending | pending | pending |
| reflex-1 | typed-decisions | 0 completed | pending | pending | pending | pending |
| reflex-1 | BANKING77 | 0 completed | pending | pending | pending | pending |
| reflex-1 | CLINC150 in-domain | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | typed-decisions | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | BANKING77 | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | CLINC150 in-domain | 0 completed | pending | pending | pending | pending |

CPU-only 4 threads, Q4_K_M, runtime build11443, USB powered. Phone ctx2048, general ctx4096. Warm timing excludes first request and includes ADB/HTTP. General per-question accuracy and exact-record metrics are separate; CLINC excludes OOS. Raw Qwen uses fixed no-chat-template JSON-grammar treatment; not a claim about every Qwen prompting/runtime strategy.

| Model | A32 cohort | Battery min–max | AP max | Android thermal status max | Sampled max PSS | RSS high-water |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone180 | 28.90–37.50 °C | 52.60 °C | 3.00 | 1090.65 MiB | 1093.00 MiB |

PSS maxima are sampled; RSS high-water is kernel-reported. USB power precludes an energy-saving claim.

Probability calibration (explicit hard labels, successful decisions only; failures still count in accuracy):

```json
[
  {
    "directory": "run-01",
    "metrics": {
      "valid_decisions": 180,
      "ece_10_equal_width_bins": 0.03097985240746409,
      "mean_multiclass_brier_sum": 0.1338160429011829,
      "hard_label_nll": 0.3177925623799157,
      "bins": [
        {
          "bin": 2,
          "n": 1,
          "mean_selected_probability": 0.23801604912448804,
          "accuracy": 1
        },
        {
          "bin": 4,
          "n": 5,
          "mean_selected_probability": 0.4624825915685358,
          "accuracy": 0
        },
        {
          "bin": 5,
          "n": 7,
          "mean_selected_probability": 0.5544954914123206,
          "accuracy": 0.5714285714285714
        },
        {
          "bin": 6,
          "n": 6,
          "mean_selected_probability": 0.6402181918320586,
          "accuracy": 0.6666666666666666
        },
        {
          "bin": 7,
          "n": 7,
          "mean_selected_probability": 0.7619083210773232,
          "accuracy": 0.8571428571428571
        },
        {
          "bin": 8,
          "n": 11,
          "mean_selected_probability": 0.8576935980211633,
          "accuracy": 0.8181818181818182
        },
        {
          "bin": 9,
          "n": 143,
          "mean_selected_probability": 0.9868775019916952,
          "accuracy": 0.9790209790209791
        }
      ],
      "definitions": "Explicit hard gold labels; Brier=sum over options, not divided by class count; NLL clipped at1e-12; failures excluded from calibration only, never accuracy."
    }
  }
]
```

Resource traces and measured sampled maxima:

```json
[
  {
    "directory": "run-01",
    "resources": {
      "max_android_thermal_status": 3,
      "battery_c_min": 28.9,
      "battery_c_max": 37.5,
      "ap_c_max": 52.6,
      "skin_c_max": 42.6,
      "sampled_max_pss_kib": 1116821,
      "rss_high_water_kib": 1119236
    }
  }
]
```

Paired fresh180 comparisons (reflex minus comparator):

```json
{}
```
