import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_question_id(question_id: Any) -> Any:
    if isinstance(question_id, int):
        return question_id
    if isinstance(question_id, str):
        if question_id.isdigit():
            return int(question_id)
        return question_id.strip()
    return question_id


def find_rollout_dirs(root: Path, branch_name: str) -> List[str]:
    branch = root / branch_name
    if not branch.exists() or not branch.is_dir():
        return []

    rollout_names: List[str] = []
    for child in branch.iterdir():
        if not child.is_dir():
            continue
        if re.fullmatch(r"rollout_\d+", child.name):
            rollout_names.append(child.name)
    return sorted(rollout_names, key=lambda name: int(name.split("_")[1]))


def list_jsonl_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"], key=lambda p: p.name)


def question_key(record: Dict[str, Any]) -> Any:
    return normalize_question_id(record.get("question_id") or record.get("id"))


def build_pair(yes_record: Dict[str, Any], no_record: Dict[str, Any], rollout: str) -> Dict[str, Any]:
    question_id = yes_record.get("question_id", no_record.get("question_id", yes_record.get("id", no_record.get("id"))))
    q1_answer = yes_record.get("answer")
    q2_answer = no_record.get("answer")

    return {
        "rollout": rollout,
        "question_id": question_id,
        "model": yes_record.get("model") or no_record.get("model"),
        "language": yes_record.get("language") or no_record.get("language"),
        "q1_str": yes_record.get("response") or yes_record.get("original_response") or yes_record.get("q_str"),
        "q2_str": no_record.get("response") or no_record.get("original_response") or no_record.get("q_str"),
        "q1_answer": q1_answer,
        "q2_answer": q2_answer,
    }


def merge_rollout(yes_paths: List[Path], no_paths: List[Path], rollout: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    yes_records: List[Dict[str, Any]] = []
    no_records: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for path in yes_paths:
        yes_records.extend(load_jsonl(path))
    for path in no_paths:
        no_records.extend(load_jsonl(path))

    yes_index: Dict[Any, Dict[str, Any]] = {}
    no_index: Dict[Any, Dict[str, Any]] = {}

    for i, rec in enumerate(yes_records):
        key = question_key(rec)
        if key in yes_index:
            warnings.append(f"Duplicate YES question_id {key} in rollout {rollout} (file {yes_paths[0].name})")
        yes_index[key] = rec

    for i, rec in enumerate(no_records):
        key = question_key(rec)
        if key in no_index:
            warnings.append(f"Duplicate NO question_id {key} in rollout {rollout} (file {no_paths[0].name})")
        no_index[key] = rec

    common_keys = sorted(set(yes_index) & set(no_index), key=lambda k: (isinstance(k, int), k))
    missing_yes = sorted(set(no_index) - set(yes_index), key=lambda k: (isinstance(k, int), k))
    missing_no = sorted(set(yes_index) - set(no_index), key=lambda k: (isinstance(k, int), k))

    for key in missing_yes:
        warnings.append(f"Missing YES record for question_id {key} in rollout {rollout}")
    for key in missing_no:
        warnings.append(f"Missing NO record for question_id {key} in rollout {rollout}")

    merged: List[Dict[str, Any]] = []
    for key in common_keys:
        merged.append(build_pair(yes_index[key], no_index[key], rollout))

    return merged, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge gemma4_e4b yes/no rollouts into paired JSONL records")
    parser.add_argument(
        "--language",
        choices=["english", "german"],
        default="english",
        help="Language dataset branch to merge (english or german)",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root directory containing yes/ and no/ rollout subdirectories. If omitted, derived from --language",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write merged rollout JSONL files. If omitted, derived from --language",
    )
    parser.add_argument(
        "--yes-branch",
        default="yes",
        help="Name of the yes branch directory under the input root",
    )
    parser.add_argument(
        "--no-branch",
        default="no",
        help="Name of the no branch directory under the input root",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root) if args.input_root else Path(f"{args.language}_unfaithful_cot_outputs/gemma4_e4b")
    output_root = Path(args.output_dir) if args.output_dir else Path(f"{args.language}_unfaithful_cot_eval_results/gemma4_e4b/merged")

    yes_rollouts = find_rollout_dirs(input_root, args.yes_branch)
    no_rollouts = find_rollout_dirs(input_root, args.no_branch)

    if not yes_rollouts and not no_rollouts:
        raise SystemExit(f"No rollout directories found under {input_root}/{args.yes_branch} or {input_root}/{args.no_branch}")

    common_rollouts = [r for r in yes_rollouts if r in no_rollouts]
    missing_yes_rollouts = [r for r in no_rollouts if r not in yes_rollouts]
    missing_no_rollouts = [r for r in yes_rollouts if r not in no_rollouts]

    if missing_yes_rollouts:
        print(f"Warning: no matching YES rollout directories found for: {', '.join(missing_yes_rollouts)}")
    if missing_no_rollouts:
        print(f"Warning: no matching NO rollout directories found for: {', '.join(missing_no_rollouts)}")

    if not common_rollouts:
        raise SystemExit("No matching rollout directories found in both yes/ and no/ branches.")

    print(f"Found rollouts: {', '.join(common_rollouts)}")

    for rollout in common_rollouts:
        yes_dir = input_root / args.yes_branch / rollout
        no_dir = input_root / args.no_branch / rollout
        yes_files = list_jsonl_files(yes_dir)
        no_files = list_jsonl_files(no_dir)

        if not yes_files:
            print(f"Skipping {rollout}: no .jsonl files found in {yes_dir}")
            continue
        if not no_files:
            print(f"Skipping {rollout}: no .jsonl files found in {no_dir}")
            continue

        merged, warnings = merge_rollout(yes_files, no_files, rollout)
        output_file = output_root / f"{rollout}.jsonl"
        write_jsonl(merged, output_file)

        print(f"Wrote {len(merged)} paired records to {output_file}")
        for warning in warnings:
            print(f"  Warning: {warning}")

    print("Merge completed.")


if __name__ == "__main__":
    main()
