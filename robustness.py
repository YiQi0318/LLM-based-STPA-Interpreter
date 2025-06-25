import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Config ---
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)
INPUT_DIR = "./LLMexplainer/imageblur/input_folder"  # Directory of original images
UCA_LIST_FILE = "./LLMexplainer/uca_list.txt"  # File with 1 UCA per line
OUTPUT_CSV = "blur_vs_uca_retention_batch.csv"
BLUR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
MAX_KERNEL = 21
COSINE_THRESHOLD = 0.3

# --- Load UCA list ---
with open(UCA_LIST_FILE, 'r') as f:
    uca_list = [line.strip() for line in f if line.strip()]
total_ucas = len(uca_list)

uca_embeddings = [
    client.embeddings.create(
        model="text-embedding-3-large",
        input=[uca]
    ).data[0].embedding for uca in uca_list
]

# --- Get embedding ---
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return response.data[0].embedding

# --- Blur helper ---
def apply_blur(image_array, percentage):
    ksize = int((percentage / 100.0) * MAX_KERNEL)
    ksize = max(1, ksize | 1)
    return cv2.GaussianBlur(image_array, (ksize, ksize), 0)

# --- Encode image to base64 ---
def image_pil_to_base64(image):
    import base64
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# --- LLM to extract evidence text ---
def extract_evidence_from_image(image_pil):
    prompt = "Describe the weather, road condition, objects, time, and traffic signal from this image."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a safety scenario interpreter."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_pil_to_base64(image_pil)}}
            ]}
        ]
    )
    return response.choices[0].message.content.strip()

# --- Main experiment ---
results = []

for image_name in os.listdir(INPUT_DIR):
    if not image_name.lower().endswith((".png", ".jpg")):
        continue

    original = Image.open(os.path.join(INPUT_DIR, image_name)).convert('RGB')
    print(f"Processing image: {image_name}")

    for level in BLUR_LEVELS:
        blurred_np = apply_blur(np.array(original), level)
        blurred_pil = Image.fromarray(blurred_np)

        try:
            evidence_text = extract_evidence_from_image(blurred_pil)
            evidence_embedding = get_embedding(evidence_text)

            retained_count = 0
            for uca_vec in uca_embeddings:
                sim = cosine_similarity([uca_vec], [evidence_embedding])[0][0]
                if sim >= COSINE_THRESHOLD:
                    retained_count += 1

            print(f"  Blur {level}% → {retained_count} UCAs retained")
            results.append({"image": image_name, "blur_level": level, "retained_ucas": retained_count})
        except Exception as e:
            print(f"  Error at blur {level}%: {e}")
            continue

# Save results
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

    # --- Plotting per image ---
    plt.figure(figsize=(12, 6))
    for img, group in results_df.groupby("image"):
        group = group.sort_values("blur_level")
        x = group["blur_level"].values
        y = group["retained_ucas"].values
        is_jpg = img.lower().endswith(".jpg")
        color = 'blue' if is_jpg else 'orange'
        label = "Town" if is_jpg else "Highway"
        plt.plot(x, y, marker='o', label=label, color=color)

        dy = np.gradient(y)
        threshold = np.mean(dy) + 2 * np.std(dy)
        for i in range(1, len(dy)):
            if dy[i] > threshold:
                plt.axvline(x=x[i], color='purple', linestyle='--', alpha=0.5)
                plt.text(x[i], y[i], f"↑ {x[i]}%", rotation=0, ha='center', fontsize=8, color='purple')
                break

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Blur Level (%)")
    plt.ylabel("# of UCAs Retained")
    plt.title("UCA Retention vs Blur Level Across Images")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("blur_vs_uca_plot.png")
    plt.show()

    # --- Average, Median, Std Curve ---
    agg = results_df.groupby("blur_level")["retained_ucas"].agg(["mean", "median", "std"]).reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(agg["blur_level"], agg["mean"], marker='o', label="Mean")
    plt.plot(agg["blur_level"], agg["median"], marker='s', linestyle='--', label="Median")
    plt.fill_between(agg["blur_level"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], color='gray', alpha=0.2, label="Std Dev")
    plt.xlabel("Blur Level (%)")
    plt.ylabel("UCAs Retained")
    plt.title("Average, Median, and Std Dev of UCAs Retained per Blur Level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("blur_vs_uca_avg_median_std.png")
    plt.show()
else:
    print("No valid results to plot. All GPT calls may have failed.")
