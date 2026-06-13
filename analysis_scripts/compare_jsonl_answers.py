import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def iter_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def compare_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for jsonl_file in iter_jsonl_files(input_path):
        with jsonl_file.open("r", encoding="utf-8") as handle:
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
                            "file_path": str(jsonl_file),
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
                f"file {item['file_path']} line {item['line_number']}: answer={item['answer']} q1_answer={item['q1_answer']}\n"
            )


def build_default_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / f"{input_path.name}_mismatches.txt"
    return input_path.with_name(f"{input_path.stem}_mismatches.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 'answer' and 'q1_answer' fields in a JSONL file or directory and report mismatched lines."
    )
    parser.add_argument("input_path", help="Path to a JSONL file or a directory containing JSONL files")
    parser.add_argument(
        "--output",
        help="Path to the mismatch report file. Defaults to <input_stem>_mismatches.txt",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output) if args.output else build_default_output_path(input_path)

    mismatches = compare_jsonl(input_path)
    write_report(mismatches, output_path)

    jsonl_files = iter_jsonl_files(input_path)
    total = 0
    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8") as handle:
            total += sum(1 for _ in handle)

    print(f"Checked: {input_path}")
    print(f"JSONL files: {len(jsonl_files)}")
    print(f"Total lines: {total}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()