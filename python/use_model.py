import numpy as np
import tensorflow.keras as keras
import time
from PIL import Image

# Load the model
loaded_model = keras.models.load_model("flowers.h5")

# Load and preprocess the input data
def preprocess_input(image):
    # Preprocess the image according to your model's requirements
    # Example: resizing, normalization, etc.
    preprocessed_image = image.resize((150, 150))
    preprocessed_image = np.array(preprocessed_image) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image

start_time_before_load = time.time()
# Load the image from file
image_path = "~/dl4j-examples-data/dl4j-examples/flower_photos/daisy/99306615_739eb94b9e_m.jpg"  # Replace with the path to your image file
image = Image.open(image_path)

# Preprocess the image
preprocessed_image = preprocess_input(image)

# Make predictions

start_time = time.time()
predictions = loaded_model.predict(preprocessed_image)

# Get the predicted class label
predicted_class_index = np.argmax(predictions)
class_labels = ["daisy","dandelion","roses", "sunflowers", "tulips" ]
predicted_class_label = class_labels[predicted_class_index]

end_time = time.time()
# Track the total training time
total_time = end_time - start_time
total_time_before_load = end_time - start_time_before_load
print(f"Total time: {total_time:.3f} seconds")
print(f"Total inc load time: {total_time_before_load:.3f} seconds")
print("Predicted class:", predicted_class_label)
