import yaml
import ollama
from pathlib import Path
from typing import List, Dict
import json
import os

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
                    'temperature': 0.0,  # For more deterministic responses
                    'num_predict': 1024
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"

def load_dataset(dataset_path: str) -> Dict:
    with open(dataset_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, output_path: str):
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

def run_experiment(dataset_path: str, models: List[Dict], output_path: str):
    dataset = load_dataset(dataset_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for model_config in models:
        model_name = model_config["name"]
        print(f"Running experiment with model: {model_name}")
        model = ModelInterface(model_name)

        responses = []
        for q in dataset["questions"]:
            prompt = q["problem"]
            response = model.generate(prompt)
            responses.append({
                "question": q["name"],
                "prompt": prompt,
                "response": response,
                "solution": q["solution"]
            })

        results["models"].append({
            "name": model_name,
            "responses": responses
        })

    save_results(results, output_path)
    print(f"Results saved to {output_path}")

# Create dataset file
dataset_content = """
params:
  description: Putnam Competition Problems
  id: filtered_putnambench
  pre-id: null
questions:
- name: putnam_2023_a1
  problem: For a positive integer $n$, let $f_n(x) = \\cos(x) \\cos(2x) \\cos(3x) \\cdots \\cos(nx)$. Find the smallest $n$ such that $|f_n''(0)| > 2023$.
  solution: Show that the solution is $n = 18$.
- name: putnam_2022_a1
  problem: Determine all ordered pairs of real numbers $(a,b)$ such that the line $y = ax+b$ intersects the curve $y = \\ln(1+x^2)$ in exactly one point.
  solution: Show that the solution is the set of ordered pairs $(a,b)$ which satisfy at least one of (1) $a = b = 0$, (2) $|a| \\geq 1$, and (3) $0 < |a| < 1$ and $b < \\log(1 + r_{-}^2) - ar_{-}$ or $b > \\log(1 + r_{+}^2) - ar_{+}$ where $r_{\\pm} = \\frac{1 \\pm \\sqrt{1 - a^2}}{a}$.
- name: putnam_2021_a1
  problem: 'A grasshopper starts at the origin in the coordinate plane and makes a sequence of hops. Each hop has length $5$, and after each hop the grasshopper is at a point whose coordinates are both integers; thus, there are $12$ possible locations for the grasshopper after the first hop. What is the smallest number of hops needed for the grasshopper to reach the point $(2021, 2021)$?'
  solution: The answer is $578$.
"""

# Write dataset to file
with open("putnam_problems.yaml", "w") as f:
    f.write(dataset_content)

# Define models to test
models = [
    {"name": "qwen3-coder:30b"},
    {"name": "gemma4:e4b"}
]

# Run experiment
run_experiment("putnam_problems.yaml", models, "output_results.yaml")
print("Experiment completed!")

### NEW

# run_experiment.py
import yaml
import ollama
from pathlib import Path
from typing import List, Dict
import json
import os

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
                    'temperature': 0.0,
                    'num_predict': 2048
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"

def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def create_structured_prompt(question_data: Dict) -> str:
    """Create a structured prompt with specific steps for the model"""
    return f"""Solve this mathematical problem step by step:

Problem: {question_data['problem']}

Instructions:
1. First, analyze the problem and identify the key mathematical concepts involved
2. State any relevant theorems, formulas, or techniques that apply
3. Work through the solution systematically with clear mathematical reasoning
4. Show all intermediate steps in your calculations
5. Verify your solution makes sense in the context of the problem

Your final answer should be formatted as:
Answer: [the numerical answer or mathematical expression]
Final Verification: [brief explanation of why this answer is correct]

Solve this problem following the above steps."""

def run_experiment(dataset_path: str, models: List[Dict], output_dir: str):
    dataset = load_dataset(dataset_path)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for model_config in models:
        model_name = model_config["name"]
        print(f"Running experiment with model: {model_name}")
        model = ModelInterface(model_name)

        responses = []
        for i, q in enumerate(dataset):
            prompt = create_structured_prompt(q)
            response = model.generate(prompt)
            
            # Extract answer from response
            answer = extract_answer(response)
            
            responses.append({
                "question_id": q.get("id", f"question_{i}"),
                "problem": q["problem"],
                "prompt": prompt,
                "response": response,
                "answer": answer,
                "solution": q.get("solution", None)
            })

        # Save results for this model in separate file
        output_path = os.path.join(output_dir, f"{model_name.replace(':', '_')}_results.json")
        model_results = {
            "model": model_name,
            "dataset": dataset_path,
            "responses": responses
        }
        save_results(model_results, output_path)
        print(f"Results saved to {output_path}")

def extract_answer(response_text: str) -> str:
    """Extract the answer from the model's response"""
    # Look for Answer: or Final Verification: sections
    answer_start = response_text.find("Answer:")
    if answer_start != -1:
        answer_text = response_text[answer_start + 7:].strip()
        return answer_text.split('\n')[0].strip()
    
    # If no Answer: found, return the entire response
    return response_text[:500] + "..." if len(response_text) > 500 else response_text

# Create sample dataset file (this can be replaced with any JSON file)
sample_dataset = [
    {
        "id": "putnam_2023_a1",
        "problem": "For a positive integer $n$, let $f_n(x) = \\cos(x) \\cos(2x) \\cos(3x) \\cdots \\cos(nx)$. Find the smallest $n$ such that $|f_n''(0)| > 2023$.",
        "solution": "Show that the solution is $n = 18$."
    },
    {
        "id": "putnam_2022_a1",
        "problem": "Determine all ordered pairs of real numbers $(a,b)$ such that the line $y = ax+b$ intersects the curve $y = \\ln(1+x^2)$ in exactly one point.",
        "solution": "Show that the solution is the set of ordered pairs $(a,b)$ which satisfy at least one of (1) $a = b = 0$, (2) $|a| \\geq 1$, and (3) $0 < |a| < 1$ and $b < \\log(1 + r_{-}^2) - ar_{-}$ or $b > \\log(1 + r_{+}^2) - ar_{+}$ where $r_{\\pm} = \\frac{1 \\pm \\sqrt{1 - a^2}}{a}$."
    },
    {
        "id": "putnam_2021_a1",
        "problem": "A grasshopper starts at the origin in the coordinate plane and makes a sequence of hops. Each hop has length $5$, and after each hop the grasshopper is at a point whose coordinates are both integers; thus, there are $12$ possible locations for the grasshopper after the first hop. What is the smallest number of hops needed for the grasshopper to reach the point $(2021, 2021)$?",
        "solution": "The answer is $578$."
    }
]

# Write sample dataset to file
with open("sample_dataset.json", "w") as f:
    json.dump(sample_dataset, f, indent=2)

# Define models to test
models = [
    {"name": "qwen3-coder:30b"},
    {"name": "gemma4:e4b"}
]

# Run experiment
run_experiment("sample_dataset.json", models, "experiment_results")
print("Experiment completed!")

[
    {
        "id": "unique_question_id",
        "problem": "Your mathematical problem here",
        "solution": "Expected solution (optional)"
    },
    ...
]