import json
import os

def run_ca(save_path="./LLMexplainer/brakedownCA/knowledge.json"):
    knowledge = {
        "knowledge": [
            "1. Camera captures the environment figure to the object detection model.",
            "2. Object detection sends the brake action command to the kinematic model.",
            "3. The kinematic model requests the brake command to the actuation system to actuate the action.",
            "4. Camera capture the environment figure to the object detection model.",
            "5. Object detection sends the steer action command to the kinematic model.",
            "6. The kinematic model requests the steer command to the actuation system to actuate the action.",
            "7. Customers send the destination command to the operator.",
            "8. Operator send the command to the software authorisation.",
            "9. Soft authorisation send the command to the hardware authorisation.",
            "10. Operator send the destination command to the path planning.",
            "11. Path planning send the action command to the knematic model.",
            "12. Software authorisation send the authorisation command to the path planning.",
            "13. Software authorisation send the authorisation command to the localisation."
        ]
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as file:
        json.dump(knowledge, file, indent=4)

    print(f"✅ Knowledge saved successfully to {save_path}")

# Still runnable as a standalone script
if __name__ == "__main__":
    run_ca()
