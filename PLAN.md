# Reflex roadmap

The product goal is an open, Turkish-first, CPU-runnable semantic decision
model: natural-language state, question, and runtime-defined options in;
bounded typed probabilities out. English is a control language.

## Milestone 0: measurement foundation

Before changing an architecture, measure each baseline separately in Turkish
and English:

- normal generative prompting;
- one-token label classification;
- direct causal-LM option-text / next-token scoring;
- BERTurk fixed-label classification;
- multilingual encoder, zero-shot NLI, and reranker baselines;
- tokenizer fertility, CPU latency, calibration, and answerability metrics.

BERTurk and XLM-R are fixed-label reference measurements, not Reflex models.
The frozen benchmark is never used for training, prompts, synthesis seeds, or
threshold selection.

## Reflex V0: small causal-LM scoring baseline

Use a small Turkish-capable causal LM (initially Qwen 0.6B--1.7B candidate
sizes) to score runtime-provided option text without generating an answer.
Independent duplicated branches are the correctness oracle. V0 establishes
how far direct logit scoring can go for latency, Turkish quality, and unseen
options. It does not claim native calibration, answerability, or a dedicated
decision architecture.

## Reflex V1: calibrated decision model

Keep the selected small backbone, then add decision-specific supervised
post-training, proper scoring losses, held-out temperature scaling, and
out-of-distribution calibration evaluation. Train an explicit answerability /
insufficient-evidence signal rather than forcing all probability mass onto the
given options. Low-confidence decisions route through the bounded fallback
gate to a larger model or human workflow.

We cannot claim to reproduce any undisclosed proprietary RLCD algorithm. Once
the open system is complete, we may study observable behavior and build an
open, independently designed approximation for comparative research. It must
be named and evaluated as our own method, not represented as a proprietary
algorithm reproduction.

## Reflex V2: native decision execution

Add dedicated decision representations or heads, shared-state multi-question
execution, branch isolation, and packed execution. Prove packed results match
the duplicated-branch oracle before using the optimization in evaluation.
`src/reflexmodels/packed_execution_contract.py` is the reusable check for
that requirement: a candidate packed executor must pass
`assert_packed_matches_oracle` on every batch in `reference_batches()` before
any packed result is trusted for evaluation.

V2 is conditional: if V1 already has near-reference decision quality, sound
calibration, useful answerability, and acceptable CPU latency, V2 complexity
must demonstrate a measurable benefit.

## Final comparative research

After the open roadmap is complete, compare Reflex behavior against public
claims and reproducible black-box observations of JEV-like systems where
permitted. The goal is to identify useful capabilities to emulate, while
documenting that architecture, data, training, and results are independent.

## Hard gates

- Report Turkish and English separately; never only a combined score.
- Do not alter test data, metrics, or splits after seeing results.
- Every run records configuration, seed, dataset version, hardware, and
  machine-readable metrics.
- Advance only when the current version improves an intended axis: unseen
  options, calibration, answerability, multi-question throughput, or CPU
  deployment.
