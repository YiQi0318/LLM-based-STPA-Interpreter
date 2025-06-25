import os
import json
from openai import OpenAI
import csv


# --- Setup ---
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

RESULT_DIR = "./logs"
# GT_DIR = "./GT_LLMSTPA"
GT_DIR = "./GT_CAST"
# OUT_CSV = "comparison_report.csv"
OUT_CSV = "cast_comparison_report.csv"

# --- GPT Prompt Template ---
COMPARE_PROMPT = """
GROUND TRUTH:
{gt}

RESULT:
{result}

Are these semantically equivalent in terms of scene, unsafe actions, and safety reasoning?
Respond in this format:
[Similarity Rating: Exact Match / Minor Differences / Major Differences / Mismatch]
[Explanation: ...]
"""

# --- GPT Function ---
def evaluate_similarity(gt_text: str, result_text: str) -> str:
    prompt = COMPARE_PROMPT.format(gt=gt_text, result=result_text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a safety-focused evaluator comparing logs."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# --- Helper ---
def read_result_log(path: str) -> str:
    with open(path, 'r') as f:
        data = json.load(f)
    parts = [data.get(k, '') for k in ["scene_description", "unsafe_control_actions", "loss_scenario_log", "safe_scenario_log"]]
    return "\n\n".join(parts)

# --- Main Comparison ---
def main():
    results = []
    gt_files = {}

    # Index GT files from all subdirectories
    for root, _, files in os.walk(GT_DIR):
        for file in files:
            if file.endswith(".txt"):
                base = os.path.splitext(file)[0]
                gt_files[base] = os.path.join(root, file)

    for file in os.listdir(RESULT_DIR):
        if not file.endswith("_log.json"):
            continue

        base = file.replace("_log.json", "")
        result_path = os.path.join(RESULT_DIR, file)
        gt_path = gt_files.get(base)

        if not gt_path or not os.path.exists(gt_path):
            print(f"Ground truth missing for {base}, skipping.")
            continue

        with open(gt_path, 'r') as f:
            gt_text = f.read()
        result_text = read_result_log(result_path)

        print(f"Comparing {base}...")
        comparison = evaluate_similarity(gt_text, result_text)

        # Extract simple fields
        rating_line = comparison.split("\n")[0]
        explanation = "\n".join(comparison.split("\n")[1:]).strip()
        similarity = rating_line.replace("[", "").replace("]", "").split(":")[-1].strip()

        results.append({"file": base, "similarity": similarity, "explanation": explanation})

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "similarity", "explanation"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Report saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
