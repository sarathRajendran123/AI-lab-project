import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Dict


LINE_RE = re.compile(r"^\s*line\s+(\d+)\s*:\s*(.+)$", flags=re.IGNORECASE)


def parse_file(path: Path):
    total = 0
    full_counter: Counter[str] = Counter()
    by_line: Dict[str, Counter[str]] = defaultdict(Counter)

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            m = LINE_RE.match(s)
            if m:
                line_no = m.group(1)
                rest = m.group(2).strip()
                key = f"line {line_no}: {rest}"
                full_counter[key] += 1
                by_line[line_no][rest] += 1
                total += 1
            else:
                # fallback: count the whole line
                full_counter[s] += 1
                total += 1

    return total, full_counter, by_line


def print_summary(total, full_counter: Counter, by_line) -> str:
    lines: list[str] = []
    lines.append(f"Total entries: {total}")
    lines.append(f"Unique lines: {len(full_counter)}\n")

    lines.append("Occurrences by exact line:")
    for line, cnt in full_counter.most_common():
        pct = cnt / total * 100 if total else 0
        lines.append(f"{line}  —  {cnt} times  ({pct:.1f}%)")

    lines.append("\nAggregated by line number:")
    for ln in sorted(by_line, key=lambda x: int(x)):
        counter = by_line[ln]
        line_total = sum(counter.values())
        lines.append(f"line {ln}: {line_total} total")
        for rest, cnt in counter.most_common():
            pct = cnt / line_total * 100 if line_total else 0
            lines.append(f"  {rest}  —  {cnt} times  ({pct:.1f}%)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize cleaned mismatch lines and counts.")
    parser.add_argument("path", help="Path to the cleaned mismatches file")
    parser.add_argument("--output", "-o", help="Write summary to this file")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    total, full_counter, by_line = parse_file(path)
    out = print_summary(total, full_counter, by_line)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(out, encoding="utf-8")
        print(f"Wrote summary to: {out_path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
