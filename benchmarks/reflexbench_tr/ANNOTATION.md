# ReflexBench-TR annotation protocol

ReflexBench-TR is frozen evaluation data. It must not be used in training,
few-shot prompts, paraphrase seeds, synthetic generation, temperature fitting,
or hyperparameter selection.

For subjective or semantic items, obtain two independent native-Turkish
annotations. Record raw agreement and Cohen’s kappa; resolve disagreements in
a documented adjudication pass. Every item must record language provenance.

The current `negation_seed.jsonl` contains hand-authored deterministic minimal
pairs. It is an evaluation seed, not a training set.
