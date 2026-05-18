import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def compare_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            answer = normalize_answer(record.get("answer"))
            q1_answer = normalize_answer(record.get("q1_answer"))

            if answer != q1_answer:
                mismatches.append(
                    {
                        "line_number": line_number,
                        "answer": answer,
                        "q1_answer": q1_answer,
                        "record": record,
                    }
                )

    return mismatches


def write_report(mismatches: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for item in mismatches:
            handle.write(
                f"line {item['line_number']}: answer={item['answer']} q1_answer={item['q1_answer']}\n"
            )


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_mismatches.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 'answer' and 'q1_answer' fields in a JSONL file and report mismatched lines."
    )
    parser.add_argument("input_path", help="Path to the JSONL file to check")
    parser.add_argument(
        "--output",
        help="Path to the mismatch report file. Defaults to <input_stem>_mismatches.txt",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output) if args.output else build_default_output_path(input_path)

    mismatches = compare_jsonl(input_path)
    write_report(mismatches, output_path)

    with input_path.open("r", encoding="utf-8") as handle:
        total = sum(1 for _ in handle)

    print(f"Checked: {input_path}")
    print(f"Total lines: {total}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()