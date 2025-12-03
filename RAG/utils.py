from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, TypeVar

import yaml

T = TypeVar("T")


# ====================================================================
# File I/O Utilities
# ====================================================================


def stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield one JSON object per line from an .jsonl/.ndjson file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load the entire JSONL file into memory."""
    return list(stream_jsonl(path))


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Persist an iterable of dictionaries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_prompts(prompts_path: Path | None = None) -> Dict[str, str]:
    """Load prompt templates from prompts.yaml."""
    if prompts_path is None:
        prompts_path = Path(__file__).resolve().parent / "prompts.yaml"
    return yaml.safe_load(prompts_path.read_text(encoding="utf-8"))


# ====================================================================
# Iteration Utilities
# ====================================================================


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yield fixed-size batches (last batch may be smaller)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ====================================================================
# Chat Template Utilities
# ====================================================================


def build_chat_prompt(
    tokenizer,
    model_id: str,
    system_msg: str,
    user_msg: str,
) -> str:
    """
    Build a chat prompt using the model's chat template.

    Args:
        tokenizer: The tokenizer with chat template
        model_id: Model identifier (used to detect model type)
        system_msg: System message content
        user_msg: User message content

    Returns:
        Formatted chat prompt string
    """
    if "Qwen" in model_id:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # GPT-style models
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="medium",
        model_identity=system_msg,
    )


def parse_llm_output(
    text: str,
    model_choice: str,
    expected_fields: List[str],
) -> Dict[str, str]:
    """
    Parse LLM output by extracting fields from structured text.

    Args:
        text: Raw LLM output text
        model_choice: "qwen" or "gpt" (determines delimiters)
        expected_fields: List of field names to extract (e.g., ["Evaluation", "Total rating"])

    Returns:
        Dictionary mapping field names to extracted values
    """
    if not text:
        return {field: "" for field in expected_fields}

    # Determine delimiters based on model
    think_token = "final<|message|>" if model_choice == "gpt" else "</think>"
    end_token = "<|return|>" if model_choice == "gpt" else "<|im_end|>"

    # Extract content after thinking phase
    if think_token not in text:
        return {field: "" for field in expected_fields}

    final = text.split(think_token)[-1]

    # Verify all expected fields are present
    for field in expected_fields:
        if f"{field}:::" not in final:
            return {field: "" for field in expected_fields}

    # Extract field values
    result = {}
    for i, field in enumerate(expected_fields):
        if i < len(expected_fields) - 1:
            # Not the last field - split by next field
            next_field = expected_fields[i + 1]
            value = (
                final.split(f"{field}:::")[-1]
                .split(f"{next_field}:::")[0]
                .strip()
            )
        else:
            # Last field - split by end token
            value = (
                final.split(f"{field}:::")[-1]
                .split(end_token)[0]
                .strip()
            )
        result[field] = value

    return result


# ====================================================================
# Type Conversion Utilities
# ====================================================================


def to_int(value: Any) -> int | None:
    """Safely convert a value to int, returning None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ====================================================================
# Incremental Writing Utilities
# ====================================================================


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl_incremental(
    records: Iterable[Dict[str, Any]],
    path: Path,
    batch_size: int = 100,
) -> None:
    """
    Write records to JSONL file incrementally (flush every batch_size records).

    Args:
        records: Iterable of dictionaries to write
        path: Output file path
        batch_size: Number of records to buffer before flushing
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, record in enumerate(records, 1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % batch_size == 0:
                handle.flush()


# ====================================================================
# Directory Setup Utilities
# ====================================================================


def setup_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create output directory structure based on config.

    Returns:
        Dictionary mapping dir names to Path objects
    """
    base_dir = Path(config["output"]["base_dir"])

    dirs = {
        "base": base_dir,
        "chunks": base_dir / config["output"]["chunks_dir"],
        "deduplication": base_dir / config["output"]["deduplication_dir"],
        "qa": base_dir / config["output"]["qa_dir"],
        "judgments": base_dir / config["output"]["judgments_dir"],
        "embeddings": base_dir / config["output"]["embeddings_dir"],
        "reranking": base_dir / config["output"]["reranking_dir"],
        "benchmarks": base_dir / config["output"]["benchmarks_dir"],
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs



