# This script replaces pyblur with OpenCV for blurring to avoid issues
import cv2
import os
from PIL import Image
import numpy as np

# Define input and output folders
input_folder = './LLMexplainer/imageblur/input_folder'  
output_folder = './LLMexplainer/imageblur/output_folder'  

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Control blur percentage by adjusting the kernel size
def apply_blur_with_percentage(image_array, percentage):
    max_kernel = 11  # Max kernel size (must be odd)
    ksize = int((percentage / 100.0) * max_kernel)
    ksize = max(1, ksize | 1)  # Ensure it's odd and >= 1
    return cv2.GaussianBlur(image_array, (ksize, ksize), 0)

# Process all images in the input folder
percentage = 75  # Example blur percentage
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        blurred_np = apply_blur_with_percentage(image_np, percentage)
        blurred_image = Image.fromarray(blurred_np)

        output_path = os.path.join(output_folder, filename)
        blurred_image.save(output_path)
        print(f"Processed and saved: {output_path}")
