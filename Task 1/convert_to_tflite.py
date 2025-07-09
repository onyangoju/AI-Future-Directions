import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('recycle_classifier.h5')

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('recycle_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite: recycle_classifier.tflite")
