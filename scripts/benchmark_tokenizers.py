"""Measure tokenizer fertility on parallel Turkish and English JSONL corpora.

Each input record must contain a non-empty ``text`` field. This script records
tokens per word, character, and sentence plus p50/p95 token expansion. It
deliberately loads tokenizers only: no model weights are downloaded.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

WORD_RE = re.compile(r"\S+", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: expected a non-empty string field 'text'")
        texts.append(text)
    if not texts:
        raise ValueError(f"{path}: contains no usable text records")
    return texts


def count_sentences(text: str) -> int:
    return max(1, sum(1 for value in SENTENCE_RE.findall(text) if value.strip()))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def token_count(tokenizer: Any, text: str) -> int:
    """Support both Transformers and the standalone Rust tokenizers API."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def summarize(tokenizer: Any, texts: Iterable[str]) -> dict[str, float | int]:
    per_word: list[float] = []
    per_character: list[float] = []
    per_sentence: list[float] = []
    expansions: list[int] = []
    for text in texts:
        tokens = token_count(tokenizer, text)
        word_count = max(1, len(WORD_RE.findall(text)))
        character_count = max(1, len(text))
        sentence_count = count_sentences(text)
        expansions.append(tokens)
        per_word.append(tokens / word_count)
        per_character.append(tokens / character_count)
        per_sentence.append(tokens / sentence_count)
    return {
        "examples": len(expansions),
        "tokens_per_word": statistics.fmean(per_word),
        "tokens_per_character": statistics.fmean(per_character),
        "tokens_per_sentence": statistics.fmean(per_sentence),
        "sequence_tokens_p50": percentile([float(value) for value in expansions], 0.50),
        "sequence_tokens_p95": percentile([float(value) for value in expansions], 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON result path")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config.get("models")
    corpora = config.get("corpora")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise ValueError("config.models must be a list of model identifiers")
    if not isinstance(corpora, dict) or set(corpora) != {"tr", "en"}:
        raise ValueError("config.corpora must contain exactly 'tr' and 'en' paths")

    try:
        from transformers import AutoTokenizer

        def load_tokenizer(model_name: str) -> Any:
            return AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=bool(config.get("trust_remote_code", False))
            )

        backend = "transformers"
    except ImportError:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise SystemExit(
                "Install either compatible transformers/tokenizers packages or tokenizers. "
                f"Original error: {error}"
            ) from error

        def load_tokenizer(model_name: str) -> Any:
            return Tokenizer.from_pretrained(model_name)

        backend = "tokenizers"

    texts_by_language = {language: read_texts(Path(path)) for language, path in corpora.items()}
    result: dict[str, Any] = {
        "task": "tokenizer_fertility",
        "run_id": f"tokenizer-fertility-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(args.config),
        "tokenizer_backend": backend,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "models": {},
    }
    for model_name in models:
        tokenizer = load_tokenizer(model_name)
        result["models"][model_name] = {
            language: summarize(tokenizer, texts)
            for language, texts in texts_by_language.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
