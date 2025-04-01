import logging
import os
import re
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor as model_processor, VisionEncoderDecoderModel as vedmodel

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Hide CUDA multi-GPU warnings
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""  # Disable GPU logs if necessary
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"  # Suppress XLA errors

# Disable warnings
warnings.filterwarnings("ignore")

# Suppress logging from transformers and absl
import transformers

transformers.logging.set_verbosity_error()

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress logging globally
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def segment_and_visualize(image_path, y_threshold=25, x_merge_threshold=15):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image. Check the file path.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    lines = []
    current_line = []
    for box in bounding_boxes:
        x, y, w, h = box
        if not current_line:
            current_line.append(box)
        else:
            median_y = np.median([b[1] for b in current_line])
            if abs(y - median_y) <= y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    if current_line:
        lines.append(current_line)

    line_images = []
    os.makedirs("temp_images", exist_ok=True)

    for i, line in enumerate(lines):
        x_min = max(0, min([b[0] for b in line]) - x_merge_threshold)
        y_min = max(0, min([b[1] for b in line]) - 5)
        x_max = min(img.shape[1], max([b[0] + b[2] for b in line]) + x_merge_threshold)
        y_max = min(img.shape[0], max([b[1] + b[3] for b in line]) + 5)

        line_img = img[y_min:y_max, x_min:x_max]
        temp_image_path = f"temp_images/line_{i}.png"
        cv2.imwrite(temp_image_path, line_img)
        line_images.append(temp_image_path)

    return line_images


def preprocess_image(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    target_height = 384  # TrOCR default height
    aspect_ratio = image.width / image.height
    new_width = int(target_height * aspect_ratio)
    image = image.resize((new_width, target_height), Image.LANCZOS)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


def recognize_text(image_paths):
    processor = model_processor.from_pretrained('ml_model')
    model = vedmodel.from_pretrained('ml_model')
    model.eval()

    # Mapping common OCR mistakes to digits
    ocr_digit_map = {
        'O': '0', 'o': '0',  # O (letter) -> 0 (zero)
        'I': '1', 'l': '1',  # I, lowercase l -> 1
        'Z': '2', 'z': '2',  # Z -> 2
        'S': '5', 's': '5',  # S -> 5
        'G': '6', 'g': '6',  # G -> 6
        'B': '8', '&': '8',  # B, & -> 8
    }

    extracted_text_lines = []
    os.makedirs("preprocessed_data", exist_ok=True)

    for image_path in image_paths:
        pixel_values = preprocess_image(image_path, processor)
        torch.save(pixel_values, os.path.join("preprocessed_data", f"{os.path.basename(image_path)}.pt"))

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Convert misread characters to correct digits
        corrected_text = ''.join(ocr_digit_map.get(char, char) for char in generated_text)

        # Keep only digits (in case any unexpected non-digit remains)
        cleaned_text = re.sub(r'\D', '', corrected_text)

        extracted_text_lines.append(cleaned_text)
        os.remove(image_path)

    return "\n".join(extracted_text_lines)
