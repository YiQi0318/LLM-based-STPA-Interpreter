import cv2
import os
import pytesseract
from PIL import Image
from openai import OpenAI

# === CONFIG ===
API_KEY = os.getenv("OPENAI_API_KEY", "your keys")
VIDEO_PATH = "./LLMexplainer/test_video.mp4"
TEMP_FRAMES_DIR = "temp_frames"
SELECTED_FRAMES_DIR = "selected_frames"
FRAME_INTERVAL_SECONDS = 4  # extract every N seconds

# === INIT GPT CLIENT ===
client = OpenAI(api_key=API_KEY)

# === STEP 1: Extract frames every N seconds ===
def extract_frames(video_path, output_dir, interval_seconds):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_seconds)

    frame_count, saved_count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            path = os.path.join(output_dir, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir)]

# === STEP 2: Basic OCR + tags (customize tags) ===
def describe_frame_with_ocr(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR error on {image_path}: {e}")
        return ""

def generate_fake_tags():  # Placeholder for rule-based tagging
    return "cars, road, traffic light"  # Could be extended with ML

# === STEP 3: Ask GPT if the frame is important ===
def ask_gpt_if_frame_is_important(tags, ocr_description):
    prompt = f"""
You are a safety analyst helping select important video frames for autonomous driving.

Here is the frame data:
- Tags: {tags}
- OCR Text: {ocr_description}

Should this frame be saved for further analysis based on these criteria?
1. It shows an intersection with a traffic light.
2. The distance between vehicles appears too small.
3. There is a potential safety concern.

Respond with just "Yes" or "No", and a short reason.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# === STEP 4: Main function ===
def run_gpt_frame_selector():
    print("🔄 Extracting frames...")
    frame_paths = extract_frames(VIDEO_PATH, TEMP_FRAMES_DIR, FRAME_INTERVAL_SECONDS)
    os.makedirs(SELECTED_FRAMES_DIR, exist_ok=True)

    print("🧠 Asking GPT to evaluate frames...")
    kept_count = 0
    for path in sorted(frame_paths):
        ocr_text = describe_frame_with_ocr(path)
        tags = generate_fake_tags()
        decision = ask_gpt_if_frame_is_important(tags, ocr_text)

        print(f"{os.path.basename(path)} → GPT says: {decision}")
        if decision.lower().startswith("yes"):
            # Copy to selected frames directory
            filename = os.path.basename(path)
            save_path = os.path.join(SELECTED_FRAMES_DIR, filename)
            os.rename(path, save_path)
            kept_count += 1

    print(f"\n✅ Done! {kept_count} frames kept in: {SELECTED_FRAMES_DIR}")

if __name__ == "__main__":
    run_gpt_frame_selector()
