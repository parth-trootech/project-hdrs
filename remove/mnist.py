import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from scipy.ndimage import rotate

# Load MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

def add_random_variations(digit):
    """
    Apply random transformations to introduce variation.
    """
    # Random rotation (-20 to 20 degrees)
    angle = random.uniform(-20, 20)
    digit = rotate(digit, angle, reshape=False, mode='nearest')

    # Random padding for shifting
    pad_x = random.randint(2, 5)  # Horizontal shift variation
    pad_y = random.randint(2, 5)  # Vertical shift variation
    digit = np.pad(digit, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=255)

    # Ensure final shape is (28, 28)
    return digit[:28, :28]

def generate_mnist_image(image_name="output.png", num_lines=2, digits_per_line=5, img_size=(300, 150)):
    """
    Generate an image with handwritten MNIST digits arranged in multiple lines with no overlapping.
    """
    fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100))
    ax.axis("off")

    # Create blank canvas
    canvas = np.ones((img_size[1], img_size[0])) * 255  # White background

    # Digit placement parameters
    spacing_y = img_size[1] // num_lines  # Fixed vertical spacing

    for i in range(num_lines):
        x_offset = random.randint(5, 15)  # Start position with randomness

        for j in range(digits_per_line):
            # Randomly select and transform a digit
            idx = np.random.randint(0, len(x_train))
            digit = add_random_variations(x_train[idx])

            digit_height, digit_width = digit.shape

            # Compute position with random shifts
            y_offset = i * spacing_y + random.randint(-5, 5)  # Small vertical jitter

            # Ensure y_offset is valid
            if y_offset + digit_height > img_size[1]:
                y_offset = img_size[1] - digit_height  # Adjust to fit within canvas

            # Ensure x_offset is within bounds
            if x_offset + digit_width > img_size[0]:
                break  # Stop placing more digits in this row if no space left

            # **Final bounds check before placement**
            if (
                0 <= x_offset < img_size[0] - digit_width and
                0 <= y_offset < img_size[1] - digit_height
            ):
                canvas[y_offset:y_offset + digit_height, x_offset:x_offset + digit_width] = digit

            # Move x_offset forward for the next digit
            x_offset += digit_width + random.randint(5, 20)  # Add random spacing

    # Save image
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# Create output directory
output_dir = "mnist_generated_images"
os.makedirs(output_dir, exist_ok=True)

# Generate multiple images
for i in range(10):  # Generate 10 images
    image_path = os.path.join(output_dir, f"mnist_image_{i}.png")
    generate_mnist_image(image_name=image_path, num_lines=2, digits_per_line=5, img_size=(300, 150))

print(f"Generated images are saved in '{output_dir}'")
