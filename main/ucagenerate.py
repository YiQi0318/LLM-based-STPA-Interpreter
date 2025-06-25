import os
import json
from openai import OpenAI

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

def generate_uca_with_hazard_links(mapped_action, hazards):
    prompt = f"""
    You are an expert in system safety analysis using STPA (System-Theoretic Process Analysis).

    Analyze this control action:
    "{mapped_action}"

    Hazards:
    """
    for hazard in hazards:
        prompt += f"- {hazard['index']}: {hazard['hazard']}\n"

    prompt += """
    Output Format:
    [UCA1] Description (H1, H2)
    [UCA2] Description (H3)
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def run_uca_generation(
    mapped_actions_path="./LLMexplainer/brakedownCA/mapped_actions.json",
    hazards_path="./LLMexplainer/ucafinder/hazard.json",
    output_path="./LLMexplainer/ucafinder/uca_with_hazards.json"
):
    with open(mapped_actions_path, "r") as f:
        mapped_data = json.load(f)

    with open(hazards_path, "r") as f:
        hazards_data = json.load(f)

    uca_results = []
    for action in mapped_data["mapped_actions"]:
        control_actions = action["mapped_action"].split("\n")[1:]

        for ca in control_actions:
            print(f"🛠️ Generating UCAs for: {ca.strip()}")
            try:
                uca_result = generate_uca_with_hazard_links(ca.strip(), hazards_data["knowledge"])
            except Exception as e:
                print(f"⚠️ Error generating UCA for '{ca}': {e}")
                uca_result = "Error generating UCA."

            uca_results.append({
                "mapped_action": ca.strip(),
                "ucas_with_hazards": uca_result
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(uca_results, file, indent=4)
    print(f"✅ UCAs with hazards saved to {output_path}")

# For standalone testing
if __name__ == "__main__":
    run_uca_generation()
