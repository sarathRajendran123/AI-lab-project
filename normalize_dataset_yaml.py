"""Normalize the dataset-style YAML files used in this workspace.

This script is meant for the malformed files where each question is stored as a
multiline single-quoted `q_str` value that breaks YAML parsing. It rewrites
those values as block scalars, which are safe to parse and preserve the text
exactly enough for downstream use.

Examples:
  python normalize_dataset_yaml.py dataset_extension/yes_eng.yaml --in-place
  python normalize_dataset_yaml.py dataset_extension/yes_eng.yaml --output fixed.yaml
  python normalize_dataset_yaml.py dataset_extension/yes_eng.yaml --target-question-key question_by_qid --in-place
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml


QUESTION_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>question_by_[A-Za-z0-9_]+):\s*$")
QUESTION_INDEX_RE = re.compile(r"^(?P<indent>\s{2})(?P<index>\d+):\s*$")
QSTR_START_RE = re.compile(r"^(?P<indent>\s{4})q_str:\s*'(?P<rest>.*)$")


def _normalize_qstr_lines(block_lines: List[str]) -> Optional[str]:
    if not block_lines:
        return None

    first_line = block_lines[0]
    match = QSTR_START_RE.match(first_line)
    if not match:
        return None

    fragments: List[str] = []
    first_fragment = match.group("rest")
    if first_fragment.endswith("'") and len(first_fragment) > 1:
        first_fragment = first_fragment[:-1]
    fragments.append(first_fragment.rstrip())

    for line in block_lines[1:]:
        if line.strip() == "":
            fragments.append("")
        elif line.startswith("      "):
            fragments.append(line[6:])
        elif line.startswith("    "):
            fragments.append(line[4:])
        else:
            fragments.append(line.lstrip())

    return "\n".join(fragments).strip("\n")


def normalize_dataset_yaml_text(raw_text: str, target_question_key: Optional[str] = None) -> str:
    lines = raw_text.splitlines()
    question_key_index: Optional[int] = None
    question_key_name: Optional[str] = None

    for index, line in enumerate(lines):
        match = QUESTION_KEY_RE.match(line)
        if match:
            question_key_index = index
            question_key_name = match.group("key")
            break

    if question_key_index is None or question_key_name is None:
        raise ValueError("Could not find a question_by_* section in the input file.")

    params_text = "\n".join(lines[:question_key_index]).rstrip()
    params_data = yaml.safe_load(params_text) if params_text else {}

    question_lines = lines[question_key_index + 1 :]
    questions: Dict[int, Dict[str, str]] = {}

    current_index: Optional[int] = None
    current_block: List[str] = []

    def flush_block() -> None:
        nonlocal current_index, current_block
        if current_index is None:
            current_block = []
            return

        q_str = _normalize_qstr_lines(current_block)
        if q_str is None:
            raise ValueError(f"Could not parse q_str for question {current_index}.")

        questions[current_index] = {"q_str": q_str}
        current_index = None
        current_block = []

    for line in question_lines:
        index_match = QUESTION_INDEX_RE.match(line)
        if index_match:
            flush_block()
            current_index = int(index_match.group("index"))
            continue

        if current_index is not None:
            current_block.append(line)

    flush_block()

    output_key = target_question_key or question_key_name
    normalized_data = {
        "params": params_data.get("params", params_data),
        output_key: questions,
    }

    return yaml.safe_dump(
        normalized_data,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize malformed dataset YAML files.")
    parser.add_argument("input", help="Path to the YAML file to normalize")
    parser.add_argument("--output", help="Write the normalized YAML to this path")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file")
    parser.add_argument(
        "--target-question-key",
        help="Rename question_by_* to this key in the output",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    normalized_text = normalize_dataset_yaml_text(
        input_path.read_text(encoding="utf-8"),
        target_question_key=args.target_question_key,
    )

    if args.in_place:
        input_path.write_text(normalized_text, encoding="utf-8")
        return

    if args.output:
        Path(args.output).write_text(normalized_text, encoding="utf-8")
        return

    print(normalized_text, end="")


if __name__ == "__main__":
    main()