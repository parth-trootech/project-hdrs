import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_and_visualize(image_path, y_threshold=25, x_merge_threshold=15):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image. Check the file path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Sort by Y-position (top to bottom)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    # Group bounding boxes into horizontal lines using hierarchical clustering
    lines = []
    current_line = []

    for box in bounding_boxes:
        x, y, w, h = box
        if not current_line:
            current_line.append(box)
        else:
            # Compare to the median Y position of the current line
            median_y = np.median([b[1] for b in current_line])
            if abs(y - median_y) <= y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

    if current_line:
        lines.append(current_line)  # Add the last detected line

    # Draw bounding boxes on the original image
    img_with_contours = img.copy()
    line_images = []

    for i, line in enumerate(lines):
        # Merge bounding boxes within the same line
        x_min = min([b[0] for b in line])
        y_min = min([b[1] for b in line])
        x_max = max([b[0] + b[2] for b in line])
        y_max = max([b[1] + b[3] for b in line])

        # Allow slight merging of close boxes
        x_min = max(0, x_min - x_merge_threshold)
        x_max = min(img.shape[1], x_max + x_merge_threshold)

        # Expand the Y-boundaries slightly to include uneven digits
        y_min = max(0, y_min - 5)
        y_max = min(img.shape[0], y_max + 5)

        # Draw rectangle
        cv2.rectangle(img_with_contours, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Crop and save
        line_img = img[y_min:y_max, x_min:x_max]
        line_images.append(line_img)
        cv2.imwrite(f"line_{i}.png", line_img)

    # Convert BGR to RGB for visualization
    img_with_contours = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)

    # Display the detected lines
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_contours)
    plt.axis("off")
    plt.title("Detected Contours (Text Lines)")
    plt.show()

    return line_images


image_path = "mnist_generated_images/mnist_image_7.png"  # Update if necessary
lines = segment_and_visualize(image_path)
