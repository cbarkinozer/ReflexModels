# Samsung A32 benchmark results

Android evaluation covers the selected model and matched cached/uncached app-router comparisons, plus a schema-format control. Beta, historical and general Android runs are deferred. General benchmark scores reported elsewhere were measured with PyTorch on GPU.

Only completed jobs have scores; pending jobs are explicit.

| Model | A32 cohort | Cases / decisions | Accuracy | Macro-F1 | Mean latency | Warm p50 / p95 | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 180 / 180 | 91.11% | 91.06% | 31.36 s | 32.34 / 33.87 s | 0 |
| Qwen app router (cache off) | Fresh phone (180) | 180 / 180 | 55.56% | 53.71% | 41.14 s | 41.05 / 44.92 s | 0 |
| Qwen app router (cache on) | Fresh phone (180) | 180 / 180 | 54.44% | 52.52% | 35.40 s | 36.31 / 38.83 s | 0 |
| reflex-1-beta | Fresh phone (180) | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1 | Historical phone (original 90) | 0 completed | deferred | deferred | deferred | deferred | deferred |
| Qwen app router (cache off) | Historical phone (original 90) | 0 completed | deferred | deferred | deferred | deferred | deferred |
| Qwen schema JSON (cache off) | Fresh phone (180) | 180 / 180 | 23.33% | 22.41% | 24.93 s | 25.44 / 26.70 s | 0 |
| Qwen schema JSON (cache on) | Fresh phone (180) | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1 | typed-decisions | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1 | BANKING77 | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1 | CLINC150 in-domain | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1-beta | typed-decisions | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1-beta | BANKING77 | 0 completed | deferred | deferred | deferred | deferred | deferred |
| reflex-1-beta | CLINC150 in-domain | 0 completed | deferred | deferred | deferred | deferred | deferred |

CPU-only 4 threads, Q4_K_M, runtime build11443, USB powered. Phone ctx2048, general ctx4096. Warm timing excludes first request and includes ADB/HTTP. General per-question accuracy and exact-record metrics are separate; CLINC excludes OOS. Raw Qwen uses fixed no-chat-template JSON-grammar treatment; not a claim about every Qwen prompting/runtime strategy.

| Model | A32 cohort | Battery min–max | AP max | Android thermal status max | Sampled max PSS | RSS high-water |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reflex-1 | Fresh phone (180) | 28.90–37.50 °C | 52.60 °C | 3.00 | 1090.65 MiB | 1093.00 MiB |
| Qwen app router (cache off) | Fresh phone (180) | 31.60–37.60 °C | 53.10 °C | 3.00 | 796.23 MiB | 838.14 MiB |
| Qwen app router (cache on) | Fresh phone (180) | 29.30–37.50 °C | 52.90 °C | 3.00 | 854.83 MiB | 896.80 MiB |
| Qwen schema JSON (cache off) | Fresh phone (180) | 31.80–37.40 °C | 53.00 °C | 3.00 | 764.54 MiB | 813.60 MiB |

PSS maxima are sampled; RSS high-water is kernel-reported. USB power precludes an energy-saving claim.

Reported token use and cache behaviour (all requests):

```json
[
  {
    "directory": "run-01",
    "usage": {
      "requests_with_input_usage": 180,
      "mean_reported_input_tokens": 456.1722222222222,
      "mean_output_tokens": 0,
      "requested_cache_values": [],
      "requests_with_cache_reporting": 0,
      "requests_reporting_nonzero_cached_tokens": 0,
      "mean_reported_cached_tokens": null,
      "cached_fraction_of_reported_input_tokens": null,
      "mean_server_prompt_evaluated_tokens": null,
      "note": "Measured server usage across all responses, including first request. Missing cache reporting means unknown, not zero. Requested flag alone does not prove reuse."
    }
  },
  {
    "directory": "qwen-legacy-fresh180",
    "usage": {
      "requests_with_input_usage": 180,
      "mean_reported_input_tokens": 657.1166666666667,
      "mean_output_tokens": 15.35,
      "requested_cache_values": [
        false
      ],
      "requests_with_cache_reporting": 180,
      "requests_reporting_nonzero_cached_tokens": 0,
      "mean_reported_cached_tokens": 0,
      "cached_fraction_of_reported_input_tokens": 0.0,
      "mean_server_prompt_evaluated_tokens": 657.1166666666667,
      "note": "Measured server usage across all responses, including first request. Missing cache reporting means unknown, not zero. Requested flag alone does not prove reuse."
    }
  },
  {
    "directory": "qwen-legacy-cached-fresh180",
    "usage": {
      "requests_with_input_usage": 180,
      "mean_reported_input_tokens": 657.1166666666667,
      "mean_output_tokens": 15.366666666666667,
      "requested_cache_values": [
        true
      ],
      "requests_with_cache_reporting": 180,
      "requests_reporting_nonzero_cached_tokens": 179,
      "mean_reported_cached_tokens": 148.86111111111111,
      "cached_fraction_of_reported_input_tokens": 0.2265368064186133,
      "mean_server_prompt_evaluated_tokens": 508.25555555555553,
      "note": "Measured server usage across all responses, including first request. Missing cache reporting means unknown, not zero. Requested flag alone does not prove reuse."
    }
  },
  {
    "directory": "qwen-fresh180",
    "usage": {
      "requests_with_input_usage": 180,
      "mean_reported_input_tokens": 357.1722222222222,
      "mean_output_tokens": 10.372222222222222,
      "requested_cache_values": [
        false
      ],
      "requests_with_cache_reporting": 180,
      "requests_reporting_nonzero_cached_tokens": 0,
      "mean_reported_cached_tokens": 0,
      "cached_fraction_of_reported_input_tokens": 0.0,
      "mean_server_prompt_evaluated_tokens": 357.1722222222222,
      "note": "Measured server usage across all responses, including first request. Missing cache reporting means unknown, not zero. Requested flag alone does not prove reuse."
    }
  }
]
```

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
    "directory": "qwen-legacy-fresh180",
    "metrics": {
      "valid_decisions": 0,
      "note": "No decision probability distribution available"
    }
  },
  {
    "directory": "qwen-legacy-cached-fresh180",
    "metrics": {
      "valid_decisions": 0,
      "note": "No decision probability distribution available"
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
    "directory": "qwen-legacy-fresh180",
    "groups": {
      "coverage": {
        "everyday": {
          "rows": 90,
          "decisions": 90,
          "correct_decisions": 54,
          "correct_records": 54,
          "accuracy": 0.6,
          "record_accuracy": 0.6
        },
        "boundary": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 21,
          "correct_records": 21,
          "accuracy": 0.4666666666666667,
          "record_accuracy": 0.4666666666666667
        },
        "indirect": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 25,
          "correct_records": 25,
          "accuracy": 0.5555555555555556,
          "record_accuracy": 0.5555555555555556
        }
      },
      "workflow": {
        "phone_routing": {
          "rows": 180,
          "decisions": 180,
          "correct_decisions": 100,
          "correct_records": 100,
          "accuracy": 0.5555555555555556,
          "record_accuracy": 0.5555555555555556
        }
      },
      "route": {
        "answer": {
          "support": 20,
          "correct": 4,
          "recall": 0.2
        },
        "get_device_info": {
          "support": 20,
          "correct": 9,
          "recall": 0.45
        },
        "get_media_info": {
          "support": 20,
          "correct": 16,
          "recall": 0.8
        },
        "get_storage_info": {
          "support": 20,
          "correct": 11,
          "recall": 0.55
        },
        "save_note": {
          "support": 20,
          "correct": 16,
          "recall": 0.8
        },
        "search_notes": {
          "support": 20,
          "correct": 8,
          "recall": 0.4
        },
        "phone_health_check": {
          "support": 20,
          "correct": 3,
          "recall": 0.15
        },
        "search_local_files": {
          "support": 20,
          "correct": 18,
          "recall": 0.9
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
    "directory": "qwen-legacy-cached-fresh180",
    "groups": {
      "coverage": {
        "everyday": {
          "rows": 90,
          "decisions": 90,
          "correct_decisions": 52,
          "correct_records": 52,
          "accuracy": 0.5777777777777777,
          "record_accuracy": 0.5777777777777777
        },
        "boundary": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 22,
          "correct_records": 22,
          "accuracy": 0.4888888888888889,
          "record_accuracy": 0.4888888888888889
        },
        "indirect": {
          "rows": 45,
          "decisions": 45,
          "correct_decisions": 24,
          "correct_records": 24,
          "accuracy": 0.5333333333333333,
          "record_accuracy": 0.5333333333333333
        }
      },
      "workflow": {
        "phone_routing": {
          "rows": 180,
          "decisions": 180,
          "correct_decisions": 98,
          "correct_records": 98,
          "accuracy": 0.5444444444444444,
          "record_accuracy": 0.5444444444444444
        }
      },
      "route": {
        "answer": {
          "support": 20,
          "correct": 5,
          "recall": 0.25
        },
        "get_device_info": {
          "support": 20,
          "correct": 8,
          "recall": 0.4
        },
        "get_media_info": {
          "support": 20,
          "correct": 17,
          "recall": 0.85
        },
        "get_storage_info": {
          "support": 20,
          "correct": 8,
          "recall": 0.4
        },
        "save_note": {
          "support": 20,
          "correct": 16,
          "recall": 0.8
        },
        "search_notes": {
          "support": 20,
          "correct": 9,
          "recall": 0.45
        },
        "phone_health_check": {
          "support": 20,
          "correct": 3,
          "recall": 0.15
        },
        "search_local_files": {
          "support": 20,
          "correct": 18,
          "recall": 0.9
        },
        "get_battery_info": {
          "support": 20,
          "correct": 14,
          "recall": 0.7
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
    "directory": "qwen-legacy-fresh180",
    "resources": {
      "max_android_thermal_status": 3,
      "battery_c_min": 31.6,
      "battery_c_max": 37.6,
      "ap_c_max": 53.1,
      "skin_c_max": 42.7,
      "sampled_max_pss_kib": 815335,
      "rss_high_water_kib": 858260
    }
  },
  {
    "directory": "qwen-legacy-cached-fresh180",
    "resources": {
      "max_android_thermal_status": 3,
      "battery_c_min": 29.3,
      "battery_c_max": 37.5,
      "ap_c_max": 52.9,
      "skin_c_max": 42.6,
      "sampled_max_pss_kib": 875343,
      "rss_high_water_kib": 918320
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
  },
  "qwen-legacy-fresh180": {
    "n": 180,
    "reflex_only_correct": 69,
    "baseline_only_correct": 5,
    "accuracy_difference": 0.35555555555555557,
    "paired_bootstrap_95ci": [
      0.2777777777777778,
      0.43333333333333335
    ],
    "exact_mcnemar_two_sided_p": 1.8345664258432237e-15,
    "bootstrap_seed": 3407,
    "bootstrap_replicates": 10000
  },
  "qwen-legacy-cached-fresh180": {
    "n": 180,
    "reflex_only_correct": 70,
    "baseline_only_correct": 4,
    "accuracy_difference": 0.36666666666666664,
    "paired_bootstrap_95ci": [
      0.28888888888888886,
      0.4444444444444444
    ],
    "exact_mcnemar_two_sided_p": 1.2898469490022716e-16,
    "bootstrap_seed": 3407,
    "bootstrap_replicates": 10000
  }
}
```
