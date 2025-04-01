import pytesseract
from PIL import Image

# # Set the path to the Tesseract executable (needed for Windows)
# # For Windows, you may need to specify the path where Tesseract is installed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust if needed

# Load the image
image_path = "/home/trootech/Documents/img.jpg"  # Path to your handwritten digits image
image = Image.open(image_path)

# Convert image to grayscale (helps improve accuracy)
image = image.convert('L')

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(image, config='--psm 10')

# Print the predicted digits
print("Predicted digits:", text.strip())
