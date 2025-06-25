from openai import OpenAI
import json
import os

api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

file_path = "./LLMexplainer/brakedownCA/knowledge.json"
# Load the JSON file with knowledge
with open(file_path, "r") as file:
    control_knowledge = json.load(file)

# Function to map natural language to control actions
def map_ca_via_gpt4(user_input):
    control_actions_text = "\n".join(control_knowledge["knowledge"])
    prompt = f"""
    You are an expert in control systems. Here is a list of standard control actions:

    {control_actions_text}

    The user will provide a natural language description of an action. Your task is to map the input description to the most relevant control action(s) from the list above. Provide the action number(s) and corresponding description(s).

    User Input: "{user_input}"

    For example:
    User Input: The AUV is braking when traffic light from yellow to red.
    Mapped action: 1. Camera captures the environment figure to the object detection model.
                    2. Object detection sends the brake action command to the kinematic model.
                    3. The kinematic model requests the brake command to the actuation system to actuate the action.

    """
    #### due to [context]  It is important to clarify that this content pertains to the scenario or environment where the unsafe control action originated, rather than the situation in which the control action was carried out.
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

# Set the folder and file path
output_folder = "./LLMexplainer/brakedownCA"
output_filename = "mapped_actions.json"
output_path = os.path.join(output_folder, output_filename)

# Ensure the folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to save mapped actions to a JSON file
def save_mapped_actions(input_text, mapped_action, output_path):
    # Load existing file if it exists, otherwise create a new one
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            data = json.load(file)
    else:
        data = {"mapped_actions": []}

    # Append the new mapping
    # 
    data = {
        "mapped_actions": [
            {
                "user_input": input_text,
                "mapped_action": mapped_action
            }
        ]
    }

    # Save back to the file
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Mapped action saved to {output_path}")

# Example usage
if __name__ == "__main__":
    user_input = "The AUV is braking when traffic light from yellow to red"
    
    # Get the mapping
    mapping_result = map_ca_via_gpt4(user_input)

    # Print the result
    print("Breakdown Control Actions:")
    print(mapping_result)
    
    # Save the result to JSON file
    save_mapped_actions(user_input, mapping_result, output_path)