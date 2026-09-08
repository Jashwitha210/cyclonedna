from PIL import Image
from src.cyclone_model import analyze_satellite_image

# Load a test satellite image
image = Image.open("data/processed/satellite.jpg")

# Send the image to the vision model
result = analyze_satellite_image(image)

# Display the result
print("Model output shape:", result.shape)
print("Image successfully analyzed!")