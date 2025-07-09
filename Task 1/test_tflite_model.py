import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="recycle_classifier.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess a test image
img_path = 'dataset/recyclable/plastic_bottle_1.jpg'
img = Image.open(img_path).resize((64, 64))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print prediction
confidence = float(output_data[0][0])
label = "Recyclable" if confidence < 0.5 else "Non-Recyclable"  # based on binary label order
print(f"Prediction: {label} (Confidence: {confidence:.4f})")
