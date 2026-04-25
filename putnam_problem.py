import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import ollama
import yaml


class ModelInterface:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.0,
                    "num_predict": 2048,
                    "stop": ["</critical_steps>"],
                },
            )
            return response["response"]
        except Exception as e:
            return f"Error generating response: {str(e)}"


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    if isinstance(data, dict) and "questions" in data:
        questions = []
        for item in data["questions"]:
            questions.append(
                {
                    "id": item.get("id") or item.get("name"),
                    "problem": item.get("problem"),
                    "solution": item.get("solution"),
                }
            )
        return questions

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported dataset format. Expected a list or a dict with 'questions'.")


def save_results(results: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def extract_last_tag_content(response_text: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    matches = re.findall(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return ""
    return matches[-1].strip()


def build_steps_context(question_data: Dict[str, Any]) -> str:
    # Preferred dataset fields: "all_steps" or "steps" as a list of strings.
    all_steps = question_data.get("all_steps") or question_data.get("steps")
    if isinstance(all_steps, list) and all_steps:
        lines = []
        for idx, step_content in enumerate(all_steps, start=1):
            lines.append(f"<step-{idx}>\n{str(step_content).strip()}\n</step-{idx}>")
        return "\n\n".join(lines)

    # Optional fallback for a single preformatted string field.
    steps_text = question_data.get("steps_text")
    if isinstance(steps_text, str) and steps_text.strip():
        return steps_text.strip()

    return "<step-1>\nNo reasoning steps provided in dataset entry.\n</step-1>"

# wrote my own prompt, because i literally can not find theirs, i have however, found the prompts for their autoraters... of which they used a lot...

#likewise... the file "extract_putnams_with_clear_answers" was moved elsewhere, and is apparently not used anymore? 
def create_structured_prompt(question_data: Dict[str, Any]) -> str:
    problem_text = str(question_data.get("problem", "")).strip()
    solution_text = str(question_data.get("solution", "")).strip()
    steps_context = build_steps_context(question_data)

    return f"""You are a math expert and you are supposed to answer these putnam bench questions, to be more specific, you do have the solution already attached, however you now need to explain to your students how to reach that answer correctly, step by step carefully, with clear and good reasoning. Do not skip any critical steps that are necessary to reach the answer. A critical step is one that establishes a key fact, insight, or result that is necessary for the final solution, or performs a calculation etc. directly used in the final solution. Include answering the final solution as a critical step.

The problem is in between <problem> and </problem> tags.

Format your answer like this:
All steps that are used for your reasoning or your calculation should be in: <all steps> and </all steps> tags.
Each step should be in its own tags like <step-1>...</step-1>, <step-2>...</step-2> and so on, in the order you think about them. Do not skip any steps, and do not combine multiple steps into one. Each step should be a single insight, calculation, or logical deduction.
It is extremely important to follow the format strictly, as we will be parsing your answer based on these tags. Do not add any extra tags or text outside of the specified format.

<problem>
{problem_text}
</problem>

<solution>
{solution_text}
</solution>
"""



### this is something i found in their repo, however, the repo is horribly structured and there is far too many files, with so many things, I would probably need literally a week constantly looking through to actually understand what they do where... and we don't have that much time, so just shortcut it...

# """We need to identify which steps in this mathematical solution are critical to reaching the answer. A critical step is one that establishes a key fact, insight, or result that is necessary for the final solution, or performs a calculation etc. directly used in the final solution. Include answering the final solution as a critical step.

# The problem is in between <problem> and </problem> tags, and all the steps are in between <all steps> and </all steps> tags.

# Please identify the steps that form the critical path to the solution. Ignore steps that:
# - Only check work without changing the path
# - Make observations that aren't used later
# - Explore dead ends
# - Restate previous results without adding new insights

# List ONLY the step numbers that are critical, in the order they build to the solution. Format your answer like this:
# <critical_steps>1,4,7,8</critical_steps> -- we will only read the last instance of <critical_steps>...</critical_steps> for your answer, so ensure you put the answer in these tags at the end of your response.

# Make sure you first think carefully about the logical dependencies between steps and what is truly necessary to establish the result, before jumping to an answer.

# Do not miss any steps out that will lead the rest of the steps to make no sense on their own. This is a hard problem, so think hard first before answering.

# <problem>
# {problem_text}
# </problem>

# <solution>
# {solution_text}
# </solution>

# <all steps>
# {steps_context}
# </all steps>"""

def extract_answer(response_text: str) -> str:
    critical_steps_text = extract_last_tag_content(response_text, "critical_steps")
    if critical_steps_text:
        return critical_steps_text

    # No valid critical_steps tag found.
    return ""


def extract_critical_steps(response_text: str) -> List[int]:
    last_value = extract_last_tag_content(response_text, "critical_steps")
    if not last_value:
        return []

    numbers = re.findall(r"\d+", last_value)
    return [int(n) for n in numbers]


def parse_models(model_names: List[str]) -> List[Dict[str, str]]:
    return [{"name": model_name.strip()} for model_name in model_names if model_name.strip()]


def run_experiment(dataset_path: str, models: List[Dict[str, str]], output_dir: str) -> None:
    dataset = load_dataset(dataset_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for model_config in models:
        model_name = model_config["name"]
        print(f"Running experiment with model: {model_name}")
        model = ModelInterface(model_name)

        responses: List[Dict[str, Any]] = []
        for i, q in enumerate(dataset):
            prompt = create_structured_prompt(q)
            response = model.generate(prompt)
            answer = extract_answer(response)
            critical_steps = extract_critical_steps(response)

            responses.append(
                {
                    "id": q.get("id", f"question_{i}"),
                    "prompt": prompt,
                    "response": response,
                    "answer": answer,
                    "critical_steps": critical_steps,
                    "solution": q.get("solution", None),
                }
            )

        output_path = str(Path(output_dir) / f"{model_name.replace(':', '_')}_results.json")
        model_results = {
            "model": model_name,
            "dataset": dataset_path,
            "responses": responses,
        }
        save_results(model_results, output_path)
        print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Putnam prompting experiments.")
    parser.add_argument("--dataset", default="sample_dataset.json", help="Path to dataset file (.json/.yaml/.yml)")
    parser.add_argument("--output-dir", default="experiment_results", help="Directory to save model outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-coder:30b", "gemma4:e4b"],
        help="One or more Ollama model names",
    )

    args = parser.parse_args()
    models = parse_models(args.models)
    run_experiment(args.dataset, models, args.output_dir)
    print("Experiment completed!")


if __name__ == "__main__":
    main()
