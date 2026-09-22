# ReflexModels — V1 Implementation Plan

ReflexModels builds small, open semantic decision models for **fast, typed, calibrated decisions without autoregressive text generation**.

The project is bilingual from the beginning:

* **Turkish = primary specialization**
* **English = control, benchmark and transfer language**

The main research question is:

> Can a small pretrained LM be converted into a non-generative decision model that handles arbitrary natural-language questions and options, performs well in Turkish and English, remains calibrated, and runs efficiently on CPU?

The project succeeds only if it provides value beyond ordinary fixed-label classifiers.

---

## 0. Core product distinction

ReflexModels is **not** primarily a fixed-label classifier.

Its core capability is:

```text
state
+
natural-language question
+
runtime-defined natural-language options
↓
probabilities
```

Example:

```python
Choice(
    state="Kullanıcı config.yaml dosyasını incelemek istiyor.",
    question="Hangi araç kullanılmalı?",
    options={
        "filesystem": "Yerel dosyaları oku",
        "web": "İnternette ara",
        "shell": "Komut çalıştır"
    }
)
```

The options may be completely new at inference time.

No retraining should be required when:

* new tools are added
* labels are renamed
* descriptions change
* a new taxonomy is supplied
* the question itself changes

This is the primary differentiator from BERT-style classifiers.

---

# 1. V1 success hypothesis

The strongest V1 claim should be:

> ReflexModels can generalize to unseen natural-language choices and tool descriptions while maintaining competitive accuracy, calibration and CPU efficiency in Turkish and English.

Secondary hypothesis:

> A ~0.6B–1.7B specialized ReflexModel can approach the bounded-decision quality of a generic ~4B model.

Do not define success as:

> Qwen3-4B model performs like Qwen3-4B.

The 4B model is the reference/prototyping model.

The small model is the actual target.

---

# 2. Model candidates

## Reference backbone

Use:

```text
Qwen3-4B-Base
```

Purpose:

* validate the architecture
* establish an upper-quality reference
* compare decision-specific training with ordinary LM behavior

## Small-model candidates

Evaluate actual available models such as:

```text
Qwen3-0.6B-Base
Qwen3-1.7B-Base
Gemma-class ~1B candidate
Llama-class ~1B candidate
```

Do not select the final small backbone based only on parameter count.

First benchmark Turkish tokenizer efficiency.

---

# 3. Tokenizer fertility benchmark

Before choosing the small model, create:

```text
scripts/benchmark_tokenizers.py
```

Corpus:

* native Turkish conversational text
* Turkish support requests
* news / formal text
* developer prompts
* mixed Turkish-English technical text

Measure:

```text
tokens / word
tokens / character
tokens / sentence
p50 / p95 sequence expansion
```

Also run the same corpus in English.

Tokenizer efficiency matters because longer Turkish tokenization directly increases CPU prefill cost.

Selection criteria:

```text
semantic quality
+
license
+
Turkish fertility
+
runtime support
+
parameter count
```

Record the decision in:

```text
DECISIONS.md
```

---

# 4. Baselines — BOTH languages from day one

Every relevant baseline must run on Turkish and English before architecture experiments begin.

Do not build English-only baselines.

---

## Baseline A — Generative LLM

Same backbone with normal instruction prompting.

Example:

```text
Return one of:
A
B
C
```

Also test JSON.

Measure:

* accuracy
* macro-F1
* latency
* generated tokens
* malformed outputs
* calibration where possible

---

## Baseline B — One-token LM classification

Use:

```text
A / B / C / D
```

and inspect token probabilities.

---

## Baseline C — Direct LM-logit scoring

No generation.

Measure candidate scores directly from the LM head.

This is the main LLM baseline.

---

# 5. Encoder baselines

These are mandatory.

## Fixed-label classifier

For Turkish:

```text
BERTurk
```

Also test an appropriate multilingual encoder:

```text
XLM-R
mDeBERTa-class encoder
```

Fine-tune them normally on fixed-label tasks.

This establishes the CPU-efficiency ceiling for conventional classification.

ReflexModels does **not** need to beat BERTurk on every fixed taxonomy.

If the taxonomy never changes, BERTurk may simply be the correct tool.

---

## Zero-shot / dynamic-label encoder

Add an NLI-style multilingual baseline:

```text
text
+
candidate description
→ entailment / compatibility score
```

This is a much closer competitor.

---

## Multilingual reranker baseline

Include:

```text
bge-reranker-v2-m3
```

Input:

```text
context/question
+
candidate
```

Output:

```text
scalar relevance score
```

This baseline already implements much of the conceptual:

```text
natural language candidate → score
```

interface.

ReflexModels must justify itself against it.

---

# 6. Headline evaluation axes

Do not organize the paper primarily around fixed-label accuracy.

Use four categories.

## A. Seen labels

Labels appeared during training.

## B. Unseen descriptions

Semantic classes are familiar but descriptions are rewritten.

Example:

```text
"web search"
```

becomes:

```text
"Search external internet resources for current information."
```

## C. Unseen labels/tools

Completely new candidate set.

Example:

Training:

```text
filesystem
web
email
```

Testing:

```text
postgres
calendar
github
memory
```

## D. Unseen domains

Train on:

```text
support
retrieval
routing
```

Test on:

```text
document processing
agent verification
```

If ReflexModels only performs well on **A**, the project has little advantage over standard encoders.

Performance on **B/C/D** is central.

---

# 7. Reflex inference primitives

Implement:

```text
Binary
Choice
Ordinal
```

Do not use product-specific terminology from other models.

---

## Binary

Absolute probability:

```text
P(true)
```

Use sigmoid semantics.

---

## Choice

Each candidate receives an independent semantic score.

Candidate scores are normalized for the supplied set when needed.

Use softmax for relative choice probability.

---

## Ordinal

Use an ordinal objective.

Do not treat ordered classes as ordinary independent categories.

---

# 8. Decision representation

Add:

```text
<DECISION>
```

Example branch:

```text
<STATE>
...

<QUESTION>
...

<OPTION>
...

<DECISION>
```

Take the hidden representation at `<DECISION>`:

```python
h = hidden_states[decision_position]
score = decision_head(h)
```

Initial head:

```python
Linear(hidden_size, 1)
```

Do not add unnecessary MLP complexity initially.

---

# 9. V0 — duplicated-branch oracle

Implement this BEFORE custom attention.

For every candidate, construct a completely independent sequence:

```text
state
+
question
+
option
+
<DECISION>
```

For three options:

```text
run 1: state + question + A
run 2: state + question + B
run 3: state + question + C
```

This is inefficient but structurally correct.

It becomes the **correctness oracle** for all later optimized implementations.

Call this:

```text
branch_mode = "duplicated"
```

---

# 10. V1 — packed branch isolation

Only after V0 works, implement packed branch execution.

Structure:

```text
                 STATE
                   │
        ┌──────────┼──────────┐
        │          │          │
     Question A Question B Question C
        │
    ┌───┼───┐
    │   │   │
   A1  A2  A3
```

Rules:

A branch may attend to:

```text
shared state
its own question
its own option
```

It must not attend to:

```text
sibling options
other questions
```

Sibling branches should use equivalent relative positions.

---

# 11. Attention backend constraint

Do not assume all optimized attention kernels support arbitrary masks.

Start with an implementation that supports explicit masks:

```text
PyTorch SDPA
or
eager attention
or
FlexAttention
```

Do not design V1 around FlashAttention-specific assumptions.

---

# 12. Mandatory correctness test

For identical weights and inputs:

```text
duplicated branch implementation
```

and:

```text
packed branch implementation
```

must produce equivalent decision scores within numerical tolerance.

Test:

```python
assert max_abs_diff < tolerance
```

This test must run in CI.

If it fails, packed attention is incorrect.

---

# 13. CPU-runtime architectural rule

Every architecture change must preserve this execution model:

```text
shared prefix
+
independent causal continuations
```

Do NOT introduce:

* bidirectional global attention
* cross-option interaction
* arbitrary cross-branch dependencies
* mechanisms impossible to represent as independent continuations

The architecture should remain compatible with a runtime conceptually equivalent to:

```text
compute state prefix once
copy/reuse prefix state
continue N branches independently
```

This is required for CPU deployment.

---

# 14. Isolation tradeoff

Branch independence has a cost.

The model cannot naturally answer relational choices requiring direct comparison between candidates.

Example:

```text
Which option is cheaper?
Which file is more specific?
Which proposal is most detailed?
```

if those properties require comparing candidates against each other.

Create:

```text
ReflexBench-Relational
```

with examples requiring sibling comparison.

Compare:

```text
independent branch scoring
vs
joint candidate serialization
```

Document where ReflexModels fails.

Do not hide this limitation.

---

# 15. Question-order experiment

CPU prefix reuse favors:

```text
STATE → QUESTION → OPTION
```

However, this means state representations are not question-conditioned.

Run one controlled comparison against:

```text
QUESTION → STATE → OPTION
```

Measure quality.

If the difference is negligible, retain state-first for efficiency.

---

# 16. Turkish from Milestone 0

All baselines and all major evaluation scripts must support:

```text
language = "tr"
language = "en"
```

Results must always be reported separately.

Never collapse them into one number only.

Example:

```text
BANKING77_EN
MASSIVE_EN
MASSIVE_TR
ReflexBench_TR
```

---

# 17. Turkish datasets

Prioritize datasets with legitimate Turkish content.

Useful candidates include:

```text
MASSIVE tr-TR
XNLI Turkish
NLI-TR where licensing permits
Turkish moderation datasets
native Turkish intent datasets
```

MASSIVE should be especially useful because aligned English/Turkish examples allow controlled cross-lingual comparisons.

Verify every dataset license before adding it.

Maintain:

```text
DATASETS.md
```

containing:

```text
name
URL
license
languages
use: train / validation / test
translated vs native
```

---

# 18. Native vs translated Turkish

Every Turkish example must include provenance:

```python
{
    "language": "tr",
    "origin": "native" | "translated" | "synthetic",
    ...
}
```

Native Turkish test sets must contain **no machine-translated examples**.

Report results separately for:

```text
native Turkish
translated Turkish
parallel bilingual data
```

Translationese must not inflate claimed Turkish quality.

---

# 19. Turkish negation suite

Create explicit Turkish minimal pairs.

Examples:

```text
uygun / uygun değil
gerekli / gerekli değil
çalışıyor / çalışmıyor
yapmalı / yapmamalı
güvenli / güvenli değil
```

Also morphological negation:

```text
gelir / gelmez
yapar / yapmaz
kullanılmalı / kullanılmamalı
```

The benchmark should test whether tiny surface changes cause correct probability flips.

This belongs in the core Turkish evaluation.

---

# 20. Synthetic Turkish data

Synthetic generation should use structured truth first.

Pipeline:

```text
latent state
↓
known label
↓
Turkish surface realization
↓
verification/audit
```

Do not ask a model to invent both the example and its truth whenever deterministic truth is possible.

---

# 21. Paraphrase audit

For every paraphrasing pipeline:

Randomly sample outputs for human review.

Record:

```text
semantic preservation rate
label flip rate
negation error rate
```

For Turkish this is mandatory.

No dataset generator may be considered complete without this audit.

---

# 22. ReflexBench-TR

Create a frozen native Turkish benchmark.

It should NOT be generated from the training pipeline.

Initial task families:

```text
intent
tool routing
relevance
binary semantic judgment
answerability
unseen options
unseen tool descriptions
Turkish negation
code-switching
```

Start small.

A few hundred high-quality examples per important category is better than tens of thousands of synthetic examples.

---

# 23. ReflexBench-TR annotation

For human-written subjective or semantic items:

Use at least two annotators.

Report:

```text
raw agreement
Cohen's kappa
```

Resolve disagreements using a documented process.

Store annotation protocol in:

```text
benchmarks/reflexbench_tr/ANNOTATION.md
```

---

# 24. Benchmark isolation rule

ReflexBench-TR is frozen evaluation data.

It must never be used for:

* training
* few-shot prompting
* paraphrasing seeds
* synthetic-data generation
* temperature tuning
* hyperparameter selection

Agents are explicitly forbidden from moving benchmark examples into training.

---

# 25. Privacy

Do not use identifiable private customer/support data unless a compliant data process exists.

For any real Turkish support dataset:

* anonymize personal identifiers
* document provenance
* document consent/legal basis where required
* avoid committing raw sensitive text to the repository

Prefer public datasets or synthetic data for V1.

---

# 26. Training mixture

Turkish-heavy training is the default hypothesis, not a fixed truth.

Test at least:

```text
50% TR / 50% EN
70% TR / 30% EN
```

Define whether percentages refer to:

```text
examples
or
tokens
```

Prefer token-aware reporting because Turkish tokenizer fertility may differ.

Measure English degradation and Turkish improvement.

Choose the final mixture empirically.

---

# 27. Training objectives

## Binary

```text
sigmoid + BCE/log-loss
```

## Choice

```text
independent scalar scores
→ softmax
→ cross entropy
```

## Ordinal

Use:

```text
CORAL-style
or
cumulative-threshold ordinal loss
```

Keep probability semantics distinct.

Do not combine unrelated pairwise-ranking losses without a clear reason.

---

# 28. Answerability

Do not put `ABSTAIN` inside Choice.

Model separately:

```text
P(answerable)
```

A request may have:

```text
answerable = 0.21
```

even though Choice still returns conditional candidate scores.

Evaluate answerability separately.

---

# 29. Calibration

Report:

```text
NLL
Brier
ECE
reliability diagrams
```

for:

```text
TR
EN
seen domain
OOD domain
seen labels
unseen labels
```

Do not use synthetic uncertainty performance as evidence of real epistemic calibration.

---

# 30. Post-hoc calibration

Fit temperature scaling on validation data.

Never test data.

Do it independently for:

```text
full precision
INT8
INT4
```

because quantization may shift score distributions.

Report before/after results.

---

# 31. Main evaluation table

Every major release should report something resembling:

```text
                     TR Seen   TR Unseen   EN Seen   EN Unseen   CPU ms
BERTurk
XLM-R
NLI encoder
BGE reranker
Qwen JSON
Qwen logits
Reflex 4B
Reflex 1.7B
Reflex 0.6B
```

Fixed-label encoders may not support some unseen-label settings naturally.

Mark these as:

```text
N/A
```

rather than forcing an artificial comparison.

That limitation is itself part of ReflexModels' value proposition.

---

# 32. CPU-first benchmarks

From the first model, benchmark:

```text
128 tokens
512 tokens
1k tokens
2k tokens
```

against:

```text
1 question
4 questions
8 questions
16 questions
32 questions
```

Report:

```text
request latency
decision latency
decisions/sec
peak RAM
```

The final performance metric is not tokens/sec.

It is:

```text
semantic decisions / second
```

---

# 33. V1 milestones

## Milestone 0 — bilingual baselines

Implement and benchmark in BOTH Turkish and English:

* [ ] generative Qwen
* [ ] one-token Qwen
* [ ] direct-logit Qwen
* [ ] BERTurk fixed-label classifier
* [ ] XLM-R / multilingual encoder
* [ ] zero-shot NLI encoder
* [ ] multilingual reranker
* [ ] tokenizer fertility benchmark
* [ ] latency benchmark
* [ ] calibration metrics

Do not modify the Qwen architecture before these exist.

---

## Milestone 1 — duplicated Reflex model

* [ ] add `<DECISION>`
* [ ] add scalar head
* [ ] implement Binary
* [ ] implement Choice
* [ ] implement Ordinal
* [ ] run each branch independently
* [ ] establish duplicated implementation as correctness oracle
* [ ] test TR + EN

---

## Milestone 2 — branch-isolated packed execution

* [ ] implement explicit branch graph
* [ ] add block attention masks
* [ ] isolate sibling options
* [ ] isolate sibling questions
* [ ] normalize sibling positions
* [ ] verify packed == duplicated scores
* [ ] permutation tests
* [ ] question-independence tests

---

## Milestone 3 — bilingual data

* [ ] MASSIVE EN/TR
* [ ] Turkish NLI
* [ ] English NLI
* [ ] retrieval data
* [ ] Turkish tool-routing data
* [ ] English tool-routing data
* [ ] synthetic filesystem environment
* [ ] synthetic tool environment
* [ ] native/translated provenance
* [ ] generator holdouts
* [ ] Turkish negation minimal pairs
* [ ] paraphrase audit

---

## Milestone 4 — first post-training

Start with LoRA/QLoRA.

Test:

```text
decision head only
head + LoRA
```

Evaluate both languages.

Hard gate:

> Continue only if ReflexModels demonstrates a meaningful advantage over direct LM logits or dynamic encoder/reranker baselines on at least one core axis: unseen-option generalization, calibration, multi-question throughput, or deployability.

Do not require it to beat BERTurk on fixed-label Turkish intent classification.

---

## Milestone 5 — ReflexBench-TR

* [ ] define tasks
* [ ] create native Turkish items
* [ ] two annotators
* [ ] agreement measurement
* [ ] freeze benchmark
* [ ] version benchmark
* [ ] prohibit benchmark use in training
* [ ] publish annotation protocol

---

## Milestone 6 — small model

Only after 4B architecture is validated:

* [ ] select small backbone using tokenizer + license + quality results
* [ ] train 1.7B-class model
* [ ] train 0.6B-class model if justified
* [ ] compare with 4B
* [ ] measure Turkish degradation
* [ ] measure unseen-option quality
* [ ] quantize

Primary target:

> Small ReflexModel approaches 4B bounded-decision quality while providing materially better CPU economics.

---

## Milestone 7 — CPU

* [ ] INT8
* [ ] INT4
* [ ] re-fit calibration after quantization
* [ ] benchmark state-prefix reuse
* [ ] benchmark multiple branches
* [ ] measure RAM
* [ ] publish reproducible hardware configuration

---

# 34. Agent operating rules

Every coding/research agent working on this repo must follow these rules.

## No silent plan changes

If implementation requires deviation from this plan:

1. record it in `DECISIONS.md`
2. explain why
3. record expected effect
4. do not silently change evaluation methodology

---

## Never modify evaluation to improve results

Agents may not:

* move test examples into training
* change test splits after seeing results
* remove difficult examples
* change metric definitions silently
* tune on the test set

---

## Every experiment is reproducible

Each run must record:

```text
git commit
model
dataset versions
config
seed
hardware
duration
training tokens/examples
metrics
```

Store configs under:

```text
configs/
```

Results under:

```text
results/
```

Use machine-readable JSON/CSV.

---

## Resource control

Before:

```text
large checkpoint download
large dataset download
full fine-tuning
long GPU job
```

the agent should verify that the run is necessary for the current milestone.

Use the cheapest experiment capable of falsifying the hypothesis.

---

## Definition of done

A task is not complete because:

```text
code compiles
```

It is complete when:

```text
tests pass
+
numbers are recorded
+
results are reproducible
```

---

# 35. Repository additions

Add:

```text
README.md
PLAN.md
DECISIONS.md
DATASETS.md
RESULTS.md

configs/
results/
benchmarks/
```

Recommended benchmark structure:

```text
benchmarks/
├── english/
├── turkish/
├── reflexbench_tr/
└── relational/
```

---

# 36. Things explicitly deferred

Do NOT implement yet:

* RL / RLCD-like training
* pretraining from scratch
* nanochat rewrite
* custom CUDA kernels
* custom CPU kernels
* Android runtime
* multimodality
* huge synthetic corpus
* long-horizon reasoning
* generative capabilities

These only become relevant after V1 proves the decision-model hypothesis.

---

# 37. Final decision rule

ReflexModels has a reason to exist if it demonstrates:

```text
dynamic natural-language decisions
+
unseen-option generalization
+
good Turkish performance
+
calibrated probabilities
+
shared-state multi-question execution
+
CPU viability
```

It does **not** need to beat every specialized encoder on every fixed-label benchmark.

If:

```text
BERTurk > ReflexModels
```

on a static 77-class Turkish classifier, that is acceptable.

If:

```text
BERTurk requires retraining for every new taxonomy
```

while ReflexModels handles new natural-language options immediately, that is the intended distinction.

The key test is therefore:

> **Can ReflexModels behave like a programmable semantic `if` statement whose possible conditions are defined at runtime?**

That is the V1 project.
