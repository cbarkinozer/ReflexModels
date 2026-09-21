# Reflex decision protocol v1

`reflex.decision.v1` is the open, Turkish-first request/response contract for
the JEV-like capability we are building. It is deliberately a constrained
decision interface, not a chat format.

## Request

```json
{
  "protocol": "reflex.decision.v1",
  "request_id": "req-42",
  "language": "tr",
  "decision_type": "choice",
  "state": "Kullanıcı config.yaml dosyasını incelemek istiyor.",
  "question": "Hangi araç kullanılmalı?",
  "options": {
    "filesystem": "Yerel dosyaları oku",
    "web": "İnternette ara"
  }
}
```

Options are supplied at runtime. They are not a fixed classifier label set.

## Response

```json
{
  "protocol": "reflex.decision.v1",
  "request_id": "req-42",
  "decision_type": "choice",
  "language": "tr",
  "selected_option": "filesystem",
  "confidence": 0.81,
  "option_probabilities": {"filesystem": 0.81, "web": 0.19}
}
```

The response has no free-text explanation field. `selected_option` must be a
key in the request's `options`, and probabilities must cover exactly those
keys and sum to one. This prevents outputting an invented action; it does not
make every semantic choice correct, so calibration and abstention remain
separate research targets.

## Shared-state batches

One input state can carry multiple independent decisions. The batch places
`state` and `language` once at the top level; every element in `decisions` has
its own `decision_id`, type, question, and runtime candidates. This is the
wire-level basis for future shared-prefix / packed parallel inference. Each
decision still returns an independent typed distribution.

## Confidence routing

The bounded response stays intact even when a downstream system chooses not to
act on it. `fallback_gate` evaluates selected-candidate confidence, top-two
margin, and normalized entropy. It returns deterministic routing metadata;
low-confidence cases can be sent to a larger model or a human workflow.
Thresholds are chosen on validation data, never the frozen test set.
