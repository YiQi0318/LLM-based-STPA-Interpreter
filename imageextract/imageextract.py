from openai import OpenAI
import json
import pytesseract
import os
from PIL import Image

# Initialize the OpenAI client
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

# Function to generate scene analysis and hierarchical planning from figure descriptions
def analyze_image_with_gpt(image_path):

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    # Define the GPT prompt with the template
    prompt = f"""
    You are a scnarios analysis for autonomous systems.
    
    Based on the figure, extract key information and format the output according to the following template:
    
    Template for Output:
        Environment Description:
            Weather: [Weather Description]
            Time: [Time Description]
            Road: [Road Description]
            Lane Markings: [Lane Markings Description]
        
        Critical Object Identification:
            Object Name]: [Object Description]
            Category: [Object Category]
            Location Box: [Location Description]
            Traffic Light: [Traffic Light Description]
            Building: [Building Description]

        Scene analysis:
            Static Attributions: [Description]
            Motion States: [Description]
            Particular Behaviors: [Description]

        Hierarchical planning:
            Meta Actions: [Description]
            Decision Description: [Description]
            Trajectory Waypoints: [Description]

    Ensure the output is concise and strictly follows the template.

    For example:
    
    Environment Description:
        Weather: Clear, sunny day with good visibility.
        Time: Daytime with ample natural light.
        Road: A straight, two-lane road with clear markings and a sidewalk on the right side.
        Lane Markings: Double yellow centerlines separate traffic directions; white edge lines define lane boundaries.
    Critical Object Identification:
        Truck: A large commercial truck occupying the central lane directly ahead.
        Category: Truck (cargo).
        Location Box: Center of the frame, leading lane.
        Gray Vehicle: A smaller car traveling slightly ahead of the truck in the adjacent lane.
        Category: Car.
        Location Box: Right lane.
    Traffic Light: A traffic signal visible on the left side of the frame, showing a yellow light, indicating preparation for stopping.
        Building: A gray building with glass blocks is located on the right side, alongside the sidewalk.
        Category: Static object.
        Location Box: Far right.
    Scene Analysis:
        Static Attributions:
            Traffic light on the left is visible and currently yellow, suggesting caution and the need to prepare for braking.
            The gray building on the right and trees in the background are static and non-interactive elements.
    Motion States:
            Truck (Center Lane): Likely decelerating or maintaining a steady, slow speed as it approaches the intersection.
            Gray Vehicle (Right Lane): Moving slightly ahead of the truck, likely at a slow or moderate pace.
    Particular Behaviors:
            Both vehicles appear to be responding to the yellow traffic light, potentially slowing or preparing to stop.
        No pedestrians or other immediate hazards are visible.
    Hierarchical Planning:
        Meta Actions:
        Prepare to decelerate based on the yellow light ahead.
        Maintain a safe following distance from the truck in the center lane.
    Decision Description:
        Slow down to align with the changing traffic signal, ensuring readiness to stop at the intersection.
        Monitor the truck and gray vehicle for their braking patterns and adjust speed accordingly.
    Trajectory Waypoints:
        Stay centered in the current lane while gradually reducing speed.
        Approach the intersection with caution, preparing to stop at an appropriate distance if the light turns red.
        Maintain awareness of potential lateral movements from the gray vehicle in the adjacent lane.
    """

    # Send request to GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Function to process all images in a folder
def process_images_in_folder(folder_path, output_json_path):
    results = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Check if the file is an image
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        print(f"Processing image: {image_name}")
        
        # Send the image to GPT for analysis
        try:
            analysis = analyze_image_with_gpt(image_path)
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")
            analysis = "Error processing this image."

        # Append the results
        results.append({
            "image_name": image_name,
            "analysis": analysis
        })
    
    # Save all results to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Results saved to {output_json_path}")

# Main function
if __name__ == "__main__":
    # Folder containing images
    folder_path = "./LLMexplainer/imageextract/images"
    
    # Output JSON path
    output_json_path = "./LLMexplainer/imageextract/output.json"
    
    # Process images in the folder
    process_images_in_folder(folder_path, output_json_path)