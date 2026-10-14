#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 120); do
  bytes=$(stat -c %s fresh180-inputs-20261015.tar.gz 2>/dev/null || echo 0)
  if [ "$bytes" = 86313256 ]; then break; fi
  sleep 10
done
echo 'e818429ee337d22ea0da6fe3b776e4d94cbc47977856e102202ebfa5b9470709  fresh180-inputs-20261015.tar.gz' | sha256sum -c -
tar -xzf fresh180-inputs-20261015.tar.gz
cd reflex-decision
export HF_HOME=/workspace/reflex-decision/cache/huggingface
export WANDB_MODE=disabled
.venv/bin/python -m pip freeze > evaluation-environment-freeze.txt
.venv/bin/python fresh-test/evaluate_frozen.py --model runs/combined-balanced-scratch-120step-s3407/adapter --test fresh-test/test.jsonl --manifest fresh-test/freeze-manifest.json --out runs/eval-fresh180-reflex1 --source-commit 570b4b26c8f834902e05b960a010f505dbfda3c6 > fresh180-evaluation.log 2>&1
printf 'FRESH_EVALUATION_COMPLETE\n'
