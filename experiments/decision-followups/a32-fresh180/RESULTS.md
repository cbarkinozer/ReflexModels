# Samsung A32 benchmark results

Only completed jobs have scores; pending jobs are explicit.

| Model | A32 cohort | Cases / decisions | Accuracy | Macro-F1 | Warm p50 / p95 | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 180 / 180 | 91.11% | 91.06% | 32.34 / 33.87 s | 0 |
| Qwen app router (cache off) | Fresh phone (180) | 0 completed | pending | pending | pending | pending |
| Qwen app router (cache on) | Fresh phone (180) | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | Fresh phone (180) | 0 completed | pending | pending | pending | pending |
| reflex-1 | Historical phone (original 90) | 0 completed | pending | pending | pending | pending |
| Qwen app router (cache off) | Historical phone (original 90) | 0 completed | pending | pending | pending | pending |
| Qwen schema JSON (cache off) | Fresh phone (180) | 180 / 180 | 23.33% | 22.41% | 25.44 / 26.70 s | 0 |
| Qwen schema JSON (cache on) | Fresh phone (180) | 0 completed | pending | pending | pending | pending |
| reflex-1 | typed-decisions | 0 completed | pending | pending | pending | pending |
| reflex-1 | BANKING77 | 0 completed | pending | pending | pending | pending |
| reflex-1 | CLINC150 in-domain | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | typed-decisions | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | BANKING77 | 0 completed | pending | pending | pending | pending |
| reflex-1-beta | CLINC150 in-domain | 0 completed | pending | pending | pending | pending |

CPU-only 4 threads, Q4_K_M, runtime build11443, USB powered. Phone ctx2048, general ctx4096. Warm timing excludes first request and includes ADB/HTTP. General per-question accuracy and exact-record metrics are separate; CLINC excludes OOS. Raw Qwen uses fixed no-chat-template JSON-grammar treatment; not a claim about every Qwen prompting/runtime strategy.

| Model | A32 cohort | Battery min–max | AP max | Android thermal status max | Sampled max PSS | RSS high-water |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 28.90–37.50 °C | 52.60 °C | 3.00 | 1090.65 MiB | 1093.00 MiB |
| Qwen schema JSON (cache off) | Fresh phone (180) | 31.80–37.40 °C | 53.00 °C | 3.00 | 764.54 MiB | 813.60 MiB |

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
  },
  {
    "directory": "qwen-fresh180",
    "metrics": {
      "valid_decisions": 0,
      "note": "No decision probability distribution available"
    }
  }
]
```

Per-route, coverage and workflow results:

```json
[
  {
    "directory": "run-01",
    "groups": {
      "coverage": {
        "everyday": {
          "rows": 90,
          "decisions": 90,
          "correct_decisions": 89,
          "correct_records": 89,
          "accuracy": 0.9888888888888889,
          "record_accuracy": 0.9888888888888889
        },
        "boundary": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 37,
          "correct_records": 37,
          "accuracy": 0.8222222222222222,
          "record_accuracy": 0.8222222222222222
        },
        "indirect": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 38,
          "correct_records": 38,
          "accuracy": 0.8444444444444444,
          "record_accuracy": 0.8444444444444444
        }
      },
      "route": {
        "answer": {
          "support": 20,
          "correct": 17,
          "recall": 0.85
        },
        "get_device_info": {
          "support": 20,
          "correct": 17,
          "recall": 0.85
        },
        "get_media_info": {
          "support": 20,
          "correct": 19,
          "recall": 0.95
        },
        "get_storage_info": {
          "support": 20,
          "correct": 17,
          "recall": 0.85
        },
        "save_note": {
          "support": 20,
          "correct": 20,
          "recall": 1.0
        },
        "search_notes": {
          "support": 20,
          "correct": 20,
          "recall": 1.0
        },
        "phone_health_check": {
          "support": 20,
          "correct": 20,
          "recall": 1.0
        },
        "search_local_files": {
          "support": 20,
          "correct": 19,
          "recall": 0.95
        },
        "get_battery_info": {
          "support": 20,
          "correct": 15,
          "recall": 0.75
        }
      }
    }
  },
  {
    "directory": "qwen-fresh180",
    "groups": {
      "coverage": {
        "everyday": {
          "rows": 90,
          "decisions": 90,
          "correct_decisions": 17,
          "correct_records": 17,
          "accuracy": 0.18888888888888888,
          "record_accuracy": 0.18888888888888888
        },
        "boundary": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 12,
          "correct_records": 12,
          "accuracy": 0.26666666666666666,
          "record_accuracy": 0.26666666666666666
        },
        "indirect": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 13,
          "correct_records": 13,
          "accuracy": 0.28888888888888886,
          "record_accuracy": 0.28888888888888886
        }
      },
      "workflow": {
        "phone_routing": {
          "rows": 180,
          "decisions": 180,
          "correct_decisions": 42,
          "correct_records": 42,
          "accuracy": 0.23333333333333334,
          "record_accuracy": 0.23333333333333334
        }
      },
      "route": {
        "answer": {
          "support": 20,
          "correct": 11,
          "recall": 0.55
        },
        "get_device_info": {
          "support": 20,
          "correct": 0,
          "recall": 0.0
        },
        "get_media_info": {
          "support": 20,
          "correct": 3,
          "recall": 0.15
        },
        "get_storage_info": {
          "support": 20,
          "correct": 2,
          "recall": 0.1
        },
        "save_note": {
          "support": 20,
          "correct": 3,
          "recall": 0.15
        },
        "search_notes": {
          "support": 20,
          "correct": 7,
          "recall": 0.35
        },
        "phone_health_check": {
          "support": 20,
          "correct": 3,
          "recall": 0.15
        },
        "search_local_files": {
          "support": 20,
          "correct": 7,
          "recall": 0.35
        },
        "get_battery_info": {
          "support": 20,
          "correct": 6,
          "recall": 0.3
        }
      }
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
  },
  {
    "directory": "qwen-fresh180",
    "resources": {
      "max_android_thermal_status": 3,
      "battery_c_min": 31.8,
      "battery_c_max": 37.4,
      "ap_c_max": 53.0,
      "skin_c_max": 42.6,
      "sampled_max_pss_kib": 782891,
      "rss_high_water_kib": 833124
    }
  }
]
```

Paired fresh180 comparisons (reflex minus comparator):

```json
{
  "qwen-fresh180": {
    "n": 180,
    "reflex_only_correct": 125,
    "baseline_only_correct": 3,
    "accuracy_difference": 0.6777777777777778,
    "paired_bootstrap_95ci": [
      0.6,
      0.75
    ],
    "exact_mcnemar_two_sided_p": 2.0549580818052442e-33,
    "bootstrap_seed": 3407,
    "bootstrap_replicates": 10000
  }
}
```
