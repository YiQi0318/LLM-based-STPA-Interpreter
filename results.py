import os
import json
from openai import OpenAI
from PIL import Image
from typing import List, Dict

# --- Configuration ---
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)
IMG_DIR = "./LLMexplainer/dataset"  # Path to your .png/.jpg files
OUTPUT_DIR = "./LLMexplainer/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Prompt Templates ---
SCENE_TEMPLATE = """
Given an image filename "{filename}" and its general context in autonomous vehicle driving,
describe the scene in detail including weather, time, road, environment complexity, and sensor visibility risk.
"""

UCA_TEMPLATE = """
Given the filename "{filename}" and that the AV is executing the control action: "{ca}",
List unsafe control actions (UCAs) that could occur, tagged with hazard codes (e.g., H1, H2).
"""

LOSS_TEMPLATE = """
Given these UCAs:
{ucas}
Generate a loss scenario log with 3 timesteps (t0, t1, t2) where these UCAs occur and cause impact.
Then describe causality and final loss.
"""

SAFE_TEMPLATE = """
Now describe a fully mitigated safe scenario for the same situation.
Include improvements at t0, t1, t2, and describe the final safe outcome.
"""

# --- GPT Query Function ---
def gpt4_prompt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# --- Main Generation Function ---
def generate_scenario_log(filename: str) -> Dict:
    ca = "The automotive vehicle is steering."

    scene_prompt = SCENE_TEMPLATE.format(filename=filename)
    scene_description = gpt4_prompt(scene_prompt)

    uca_prompt = UCA_TEMPLATE.format(filename=filename, ca=ca)
    uca_response = gpt4_prompt(uca_prompt)

    loss_prompt = LOSS_TEMPLATE.format(ucas=uca_response)
    loss_response = gpt4_prompt(loss_prompt)

    safe_response = gpt4_prompt(SAFE_TEMPLATE)

    return {
        "filename": filename,
        "control_action": ca,
        "scene_description": scene_description,
        "unsafe_control_actions": uca_response,
        "loss_scenario_log": loss_response,
        "safe_scenario_log": safe_response
    }

# --- Run on Dataset ---
def main():
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img in image_files:
        print(f"Processing {img}...")
        result = generate_scenario_log(img)

        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img)[0]}_log.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()