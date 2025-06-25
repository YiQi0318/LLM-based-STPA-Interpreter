import os
import openai
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

# Define your UCAs and visual evidence
ucas = [
    {
        "id": "UCA1.1",
        "description": "The camera provided a blurry or poor-quality image, failing to detect the yellow light.",
        "severity": "high",
        "timing": "early",
        "impact": "delayed decision-making",
        "visual_evidence": {"timestamp": "t1", "distance_to_light": 50, "speed": 60}
    },
    {
        "id": "UCA3.2",
        "description": "The camera captured the environment too late, resulting in delayed decision-making.",
        "severity": "medium",
        "timing": "mid",
        "impact": "delayed braking",
        "visual_evidence": {"timestamp": "t1", "distance_to_light": 30, "speed": 60}
    },
    {
        "id": "UCA1.2",
        "description": "The camera failed to capture parts of the scene, missing the red light.",
        "severity": "high",
        "timing": "late",
        "impact": "missed red light",
        "visual_evidence": {"timestamp": "t2", "distance_to_light": 20, "speed": 60}
    },
    {
        "id": "UCA3.2",
        "description": "The object detection system sent the brake command too late, delaying braking.",
        "severity": "high",
        "timing": "late",
        "impact": "delayed braking",
        "visual_evidence": {"timestamp": "t2", "distance_to_light": 10, "speed": 60}
    }
]

# Define weights for each metric
weights = {
    "severity": 0.5,  # 50%
    "timing": 0.3,    # 30%
    "impact": 0.2     # 20%
}

# Define scoring for severity and timing
severity_scores = {"low": 1, "medium": 2, "high": 3}
timing_scores = {"early": 1, "mid": 2, "late": 3}

# Calculate scores for each UCA
def calculate_uca_score(uca):
    severity_score = severity_scores[uca["severity"]]
    timing_score = timing_scores[uca["timing"]]
    impact_score = 3 if "delayed braking" in uca["impact"] else 2 if "missed" in uca["impact"] else 1
    total_score = (severity_score * weights["severity"]) + (timing_score * weights["timing"]) + (impact_score * weights["impact"])
    return total_score

# Calculate contribution rate
def calculate_contribution_rate(ucas):
    total_score = sum(calculate_uca_score(uca) for uca in ucas)
    contribution_rates = []
    for uca in ucas:
        uca_score = calculate_uca_score(uca)
        contribution_rate = (uca_score / total_score) * 100
        contribution_rates.append({"id": uca["id"], "contribution_rate": contribution_rate})
    return contribution_rates

# Get contribution rates
contribution_rates = calculate_contribution_rate(ucas)

# Print results
for rate in contribution_rates:
    print(f"{rate['id']}: {rate['contribution_rate']:.2f}%")


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
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

summary = generate_summary(contribution_rates)
print("\nSummary:\n", summary)