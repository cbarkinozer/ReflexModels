# V1 decision-training data contract

V1 adds an explicit answerability target. A decision record distinguishes
"one supplied option is supported" from "the state is insufficient or none of
the options is appropriate." A restricted option softmax alone cannot learn
that distinction.

```json
{
  "id": "tr-example-001",
  "language": "tr",
  "origin": "native",
  "source": "curated-v1",
  "split": "train",
  "state": "User wants to inspect only local project files.",
  "question": "Which tool should be used?",
  "options": {"filesystem": "Read local files", "web": "Search the internet"},
  "answerable": true,
  "correct_option": "filesystem"
}
```

Actual Turkish-first records must use `language: "tr"` and preserve UTF-8
text. For an insufficient-evidence example, `answerable` is `false` and
`correct_option` is `null`. `origin` is provenance (`native`, `translated`,
or `synthetic`); `source` identifies its registered source; and `split` is
`train` or `validation`. Threshold selection uses validation data only;
frozen ReflexBench-TR remains evaluation-only.
