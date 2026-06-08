import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_question_id(qid):
    if isinstance(qid, str) and qid.startswith("question_"):
        return int(qid.split("_", 1)[1])
    if isinstance(qid, str) and qid.isdigit():
        return int(qid)
    if isinstance(qid, int):
        return qid
    raise ValueError(f"Unsupported question_id format: {qid!r}")


def build_row_map(folder: Path, rollout: str):
    source_path = folder / rollout / "gemma4_e4b_results.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing expected file: {source_path}")

    rows = {}
    for obj in read_jsonl(source_path):
        qid = obj.get("question_id")
        if qid is None:
            raise ValueError(f"Row missing question_id in {source_path}")
        normalized_qid = normalize_question_id(qid)
        rows[normalized_qid] = obj
    return rows


def collect_rollouts(base_dir: Path):
    yes_dir = base_dir / "yes"
    if not yes_dir.exists():
        raise FileNotFoundError(f"Missing yes directory: {yes_dir}")
    rollout_dirs = sorted(
        [d.name for d in yes_dir.iterdir() if d.is_dir() and d.name.startswith("rollout")]
    )
    return rollout_dirs


def normalize_pair(yes_row: dict, no_row: dict, rollout: str):
    q1_answer = yes_row.get("q1_answer")
    q2_answer = no_row.get("q1_answer", no_row.get("q2_answer"))
    qid = normalize_question_id(yes_row.get("question_id"))

    return {
        "rollout": rollout,
        "question_id": f"question_{qid}",
        "model": yes_row.get("model"),
        "language": yes_row.get("language"),
        "q1_str": yes_row.get("original_response"),
        "q1_answer": q1_answer,
        "q2_str": no_row.get("original_response"),
        "q2_answer": q2_answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Join german_unfaithful_cot_eval_results gemma4_e4b yes/no rows into cleaned JSONL output."
    )
    parser.add_argument(
        "--base-dir",
        default="german_unfaithful_cot_eval_results/gemma4_e4b",
        help="Path to the gemma4_e4b dataset root.",
    )
    parser.add_argument(
        "--output",
        default="german_unfaithful_cot_eval_results/gemma4_e4b/gemma4_e4b_joined_results.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--start-rollout",
        type=int,
        default=1,
        help="Start processing rollouts from this rollout index (inclusive).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    yes_dir = base_dir / "yes"
    no_dir = base_dir / "no"

    rollout_dirs = collect_rollouts(base_dir)
    rollout_dirs = [r for r in rollout_dirs if int(r.split("_")[-1]) >= args.start_rollout]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_pairs = []
    unmatched_no = []
    matched_count = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for rollout in rollout_dirs:
            yes_rows = build_row_map(yes_dir, rollout)
            no_rows = build_row_map(no_dir, rollout)

            for qid, yes_row in yes_rows.items():
                no_row = no_rows.get(qid)
                if no_row is None:
                    missing_pairs.append((rollout, qid))
                    continue

                normalized = normalize_pair(yes_row, no_row, rollout)
                out_fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                matched_count += 1

            for qid in no_rows:
                if qid not in yes_rows:
                    unmatched_no.append((rollout, qid))

    if missing_pairs or unmatched_no:
        print("Warning: unmatched rows found while joining yes/no files.")
        if missing_pairs:
            print(f"  yes rows without matching no rows: {len(missing_pairs)}")
        if unmatched_no:
            print(f"  no rows without matching yes rows: {len(unmatched_no)}")
    print(f"Wrote {matched_count} joined rows to {output_path}")


if __name__ == "__main__":
    main()
