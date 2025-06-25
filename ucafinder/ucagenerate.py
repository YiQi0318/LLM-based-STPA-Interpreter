import os
import re
import json
from openai import OpenAI

# Initialize the OpenAI client
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)


# Load the JSON file containing mapped actions and hazards
mapped_actions_path = "./LLMexplainer/brakedownCA/mapped_actions.json"
hazards_path = "LLMexplainer/ucafinder/hazard.json"

with open(mapped_actions_path, "r") as file:
    mapped_data = json.load(file)

with open(hazards_path, "r") as file:
    hazards_data = json.load(file)

# Function to link UCAs to hazards based on GPT output
def generate_uca_with_hazard_links(mapped_action, hazards):
    # Define the prompt to generate UCAs
    prompt = f"""
    You are an expert in system safety analysis using the STPA (System-Theoretic Process Analysis) methodology.
    
    Here is a control action:
    "{mapped_data}"
    
    Your task is to:
    1. Analyze the control action using the STPA methods.
    2. Identify possible Unsafe Control Actions (UCAs) for this control action.
    3. Link each UCA to one or more hazards described below:
    """
    # Add hazards dynamically to the prompt
    for hazard in hazards:
        prompt += f"- {hazard['index']}: {hazard['hazard']}\n"

    prompt += """
    Present the output in this format:
       - UCA 1: [Description of the unsafe control action and its potential consequences][List of hazards (e.g., H1, H2)]
       - UCA 2: [Description of the unsafe control action and its potential consequences][List of hazards (e.g., H1)]
    
    For example,

    Control action: Camera capture the environment figure to the object detection model.
    [UCA1.1] The camera provides the environment image, but it is blurry or of poor quality, causing the object detection model to fail or produce incorrect results (H1, H2).
    [UCA1.2] The camera provides an incomplete environment image, omitting critical areas or objects in the scene (H1, H2).
    [UCA1.3] The camera provides an image when the environment is not properly illuminated, leading to detection errors (H1, H2).
    [UCA2.1] The camera does not provides due to a malfunction or system error, leaving the object detection model without necessary data (H1, H2).
    [UCA3.1] The camera captures the environment before the AUV reaches the area of interest, leading to outdated or irrelevant input to the object detection model (H1, H2).
    [UCA3.2] The camera captures the environment too late, after the AUV has already passed critical points, causing delays in decision-making or execution (H1, H2).
    [UCA4.1] The camera stops capturing the environment prematurely, missing critical parts of the scene during the AUV's operation (H1, H2).
    [UCA4.2] The camera continues capturing the environment unnecessarily, consuming excessive resources such as power or processing capacity without adding value (H1, H2). 
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

# Generate UCAs with hazard links for each mapped action
uca_results = []
for action in mapped_data["mapped_actions"]:
    # Split the mapped action into individual control actions based on numbering
    control_actions = action["mapped_action"].split("\n")[1:]  # Skip the initial summary line

    # Process each control action separately
    for ca in control_actions:
        print(f"Processing control action: {ca.strip()}")
        
        # Generate UCAs and link to hazards
        uca_result = generate_uca_with_hazard_links(ca.strip(), hazards_data["knowledge"])
        
        # Append the result for this control action
        uca_results.append({
            "mapped_action": ca.strip(),
            "ucas_with_hazards": uca_result
        })

# Save the results to a JSON file
output_path = "./LLMexplainer/ucafinder/uca_with_hazards.json"
with open(output_path, "w") as file:
    json.dump(uca_results, file, indent=4)

print(f"UCAs with hazard links saved to {output_path}")
