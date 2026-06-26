import re
import sys
from collections import defaultdict

def count_by_range(file_path):
    # Define the 6 line ranges
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
    range_labels = [f"{start}-{end}" for start, end in ranges]

    # Initialize counters for both answer and q1_answer per range
    answer_counts = {label: defaultdict(int) for label in range_labels}
    q1_answer_counts = {label: defaultdict(int) for label in range_labels}

    # Regex to capture: line number, answer value, q1_answer value, and count
    # Handles varying whitespace and stops at "times"
    pattern = re.compile(r'line\s+(\d+):\s*answer=(\S+)\s+q1_answer=(\S+)\s+—\s+(\d+)\s+times')

    if not file_path:
        print("Error: No file path provided.")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    line_num = int(match.group(1))
                    ans_val = match.group(2)
                    q1_val = match.group(3)
                    count = int(match.group(4))

                    # Find which range this line belongs to
                    for i, (start, end) in enumerate(ranges):
                        if start <= line_num <= end:
                            label = range_labels[i]
                            answer_counts[label][ans_val] += count
                            q1_answer_counts[label][q1_val] += count
                            break
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # Print formatted results
    print("📊 Aggregated Counts by Line Range:\n")
    for label in range_labels:
        print(f"🔹 Range: {label}")
        print(f"   answer counts:  {dict(answer_counts[label])}")
        print(f"   q1_answer counts: {dict(q1_answer_counts[label])}")
        print("-" * 45)

if __name__ == "__main__":
    # Default to the file name mentioned in your prompt
    count_by_range("german_qwen_cleaned.txt")