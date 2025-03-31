import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

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

def recognize_text(image_paths):
    processor = TrOCRProcessor.from_pretrained('ml_model')
    model = VisionEncoderDecoderModel.from_pretrained('ml_model')

    extracted_text_lines = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        extracted_text_lines.append(generated_text)
        os.remove(image_path)  # Cleanup after processing

    return "\n".join(extracted_text_lines)

# **Test the OCR on a local image**
test_image = "mnist_generated_images/mnist_image_7.png"  # Update this path
line_image_paths = segment_and_visualize(test_image)
ocr_result = recognize_text(line_image_paths)

print("Extracted Text:\n", ocr_result)
