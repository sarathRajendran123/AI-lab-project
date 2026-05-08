import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file

print("API KEY LOADED:", os.getenv("NDIF_API_KEY") is not None)

from nnsight import LanguageModel


# =========================
# MODEL INTERFACE (NDIF)
# =========================
class ModelInterface:
    def __init__(self, model_name: str):
        print(f"Connecting to remote model: {model_name}")

        self.model = LanguageModel(
            model_name,
            device_map="auto",   # handled remotely
        )
        self.tokenizer = self.model.tokenizer

    def generate(self, prompt: str) -> str:
        try:
            with self.model.generate(
                prompt,
                max_new_tokens=512,
                temperature=0.0,
                remote=True   # 🔥 KEY: runs on NDIF servers
            ) as tracer:
                output = tracer.output

            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            return f"Error generating response: {str(e)}"


# =========================
# DATASET
# =========================
def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            import yaml
            data = yaml.safe_load(f)

    if isinstance(data, dict) and "questions" in data:
        return [
            {
                "id": item.get("id") or item.get("name"),
                "problem": item.get("problem"),
                "solution": item.get("solution"),
            }
            for item in data["questions"]
        ]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported dataset format.")


# =========================
# FILE HANDLING
# =========================
def append_jsonl_record(record: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_rollout_name(output_dir: str) -> str:
    root = Path(output_dir)
    if not root.exists():
        return "rollout_1"

    nums = [
        int(m.group(1))
        for d in root.iterdir()
        if d.is_dir()
        and (m := re.fullmatch(r"rollout_(\d+)", d.name))
    ]

    return f"rollout_{max(nums) + 1}" if nums else "rollout_1"


def rollout_output_path(output_dir: str, model_name: str, rollout_name: str) -> Path:
    rollout_dir = Path(output_dir) / rollout_name
    rollout_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 FIX: safe filename
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    return rollout_dir / f"{safe_model_name}_results.jsonl"


# =========================
# PROMPT
# =========================
def create_structured_prompt(question_data: Dict[str, Any]) -> str:
    problem_text = str(question_data.get("problem", "")).strip()
    solution_text = str(question_data.get("solution", "")).strip()

    return f"""You are a math expert.

Explain the solution step-by-step clearly and logically.

Each step must be in tags:
<step-1>...</step-1>, <step-2>...</step-2>, etc.

Do not skip steps.

<problem>
{problem_text}
</problem>

<solution>
{solution_text}
</solution>
"""


# =========================
# UTILS
# =========================
def wrap_with_all_steps_tags(response_text: str) -> str:
    if "<all steps>" in response_text.lower():
        return response_text
    return f"<all steps>\n{response_text}\n</all steps>"


# =========================
# EXPERIMENT
# =========================
def run_experiment(dataset_path: str, models: List[str], output_dir: str):
    dataset = load_dataset(dataset_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rollout_name = next_rollout_name(output_dir)
    started_at = datetime.now().isoformat(timespec="seconds")

    for model_name in models:
        print(f"\nRunning model: {model_name}")
        model = ModelInterface(model_name)

        output_path = rollout_output_path(output_dir, model_name, rollout_name)

        for i, q in enumerate(dataset):
            print(f"Processing question {i+1}/{len(dataset)}")

            prompt = create_structured_prompt(q)
            response = model.generate(prompt)

            print(f"Response length: {len(response)} chars")

            response = wrap_with_all_steps_tags(response)

            record = {
                "model": model_name,
                "dataset": dataset_path,
                "rollout": rollout_name,
                "started_at": started_at,
                "id": q.get("id", f"q_{i}"),
                "prompt": prompt,
                "response": response,
                "solution": q.get("solution"),
            }

            append_jsonl_record(record, str(output_path))

        print(f"Saved to {output_path}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="putnam_problems.yaml")
    parser.add_argument("--output-dir", default="experiment_results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.1-8B"],  # 🔥 NDIF-supported model
    )

    args = parser.parse_args()

    run_experiment(args.dataset, args.models, args.output_dir)


if __name__ == "__main__":
    main()