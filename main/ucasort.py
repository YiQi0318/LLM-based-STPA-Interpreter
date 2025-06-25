import os
import json
from openai import OpenAI

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY', 'your keys')  
client = OpenAI(api_key=api_key)

# Define scoring weights
weights = {
    "severity": 0.5,
    "timing": 0.3,
    "impact": 0.2
}
severity_scores = {"low": 1, "medium": 2, "high": 3}
timing_scores = {"early": 1, "mid": 2, "late": 3}

def calculate_uca_score(uca):
    severity_score = severity_scores.get(uca.get("severity", "medium"), 2)
    timing_score = timing_scores.get(uca.get("timing", "mid"), 2)
    impact_score = 3 if "delayed braking" in uca.get("impact", "") else 2 if "missed" in uca.get("impact", "") else 1
    return (
        severity_score * weights["severity"] +
        timing_score * weights["timing"] +
        impact_score * weights["impact"]
    )

def calculate_contribution_rate(ucas):
    total_score = sum(calculate_uca_score(uca) for uca in ucas)
    return [
        {
            "id": uca.get("id", "UCA?"),
            "description": uca["description"],
            "contribution_rate": (calculate_uca_score(uca) / total_score) * 100
        }
        for uca in ucas
    ]

def generate_summary(contribution_rates):
    prompt = "Summarize the contribution rates of the following UCAs to a collision:\n"
    for rate in contribution_rates:
        prompt += f"- {rate['id']}: {rate['contribution_rate']:.2f}%\n"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def run_uca_sort(output_path="ucasort_output.json", input_path="ucafinder/uca_with_hazards.json"):
    # Load UCAs from previous step
    with open(input_path, "r") as f:
        raw_ucas = json.load(f)

    # Extract usable UCAs from the raw file
    ucas = []
    for idx, item in enumerate(raw_ucas):
        ucas.append({
            "id": f"UCA{idx+1}",
            "description": item["ucas_with_hazards"],
            "severity": "high",  # Placeholder — could be inferred from text later
            "timing": "late",    # Same as above
            "impact": "delayed braking",  # Simplified assumption
            "visual_evidence": {}
        })

    # Score and summarize
    contribution_rates = calculate_contribution_rate(ucas)
    summary = generate_summary(contribution_rates)

    # Only make directories if needed
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "contribution_rates": contribution_rates,
            "summary": summary
        }, f, indent=4)

    # Print results
    print("\n🎯 Contribution Rates:")
    for rate in contribution_rates:
        print(f"{rate['id']}: {rate['contribution_rate']:.2f}%")

    print("\n📋 Summary:\n", summary)

# Allow CLI testing
if __name__ == "__main__":
    run_uca_sort("ucasort_output.json", "ucafinder/uca_with_hazards.json")
