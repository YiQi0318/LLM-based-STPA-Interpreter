from openai import OpenAI
import json
import os

api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

def map_ca_via_gpt4(user_input, control_knowledge):
    control_actions_text = "\n".join(control_knowledge["knowledge"])
    prompt = f"""
    You are an expert in control systems. Here is a list of standard control actions:

    {control_actions_text}

    The user will provide a natural language description of an action. Your task is to map the input description to the most relevant control action(s) from the list above. Provide the action number(s) and corresponding description(s).

    User Input: "{user_input}"
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

def save_mapped_actions(input_text, mapped_action, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "mapped_actions": [
            {
                "user_input": input_text,
                "mapped_action": mapped_action
            }
        ]
    }
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"✅ Mapped action saved to {output_path}")

def run_brakedownca(
    user_input="The AUV is braking when traffic light from yellow to red",
    knowledge_path="./LLMexplainer/brakedownCA/knowledge.json",
    output_path="./LLMexplainer/brakedownCA/mapped_actions.json"
):
    with open(knowledge_path, "r") as file:
        control_knowledge = json.load(file)

    mapping_result = map_ca_via_gpt4(user_input, control_knowledge)
    print("🔍 Breakdown Control Actions:")
    print(mapping_result)
    save_mapped_actions(user_input, mapping_result, output_path)

# If you still want to run it standalone
if __name__ == "__main__":
    run_brakedownca()