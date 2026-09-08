
import tensorflow as tf
import numpy as np
from PIL import Image


# Load pretrained vision model
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)


def analyze_satellite_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize for MobileNetV2
    image = image.resize((224, 224))

    # Convert to array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocess
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Run model
    predictions = model.predict(
        image_array,
        verbose=0
    )

    return predictions