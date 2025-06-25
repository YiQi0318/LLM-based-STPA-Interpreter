from openai import OpenAI
import json
import os
from PIL import Image

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)

def analyze_image_with_gpt(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    prompt = f"""
    You are a scenarios analysis assistant for autonomous systems.
    
    Based on the figure, extract key information and format the output according to the following template:

    [Template omitted for brevity — keep as in original]
    """

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

def process_images_in_folder(folder_path, output_json_path):
    results = []
    
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, image_name)
        print(f"📷 Processing image: {image_name}")

        try:
            analysis = analyze_image_with_gpt(image_path)
        except Exception as e:
            print(f"❌ Failed to process {image_name}: {e}")
            analysis = "Error processing this image."

        results.append({
            "image_name": image_name,
            "analysis": analysis
        })
    
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"✅ Image analysis saved to {output_json_path}")

def run_image_extraction(
    folder_path="./LLMexplainer/imageextract/images",
    output_path="./LLMexplainer/imageextract/output.json"
):
    process_images_in_folder(folder_path, output_path)

# Run standalone if needed
if __name__ == "__main__":
    run_image_extraction()
