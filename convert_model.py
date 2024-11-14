import tensorflow as tf
import argparse
import os

def load_model(model_path):
    """Load a Keras model from a given path."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def convert_model_to_tflite(model, output_path, quantization=None, input_shape=None):
    """Convert a Keras model to TensorFlow Lite format with optional quantization and save it."""
    # Initialize converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set input shape if specified (for dynamic input handling)
    if input_shape:
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        model._set_inputs(tf.TensorSpec(shape=[1] + input_shape, dtype=tf.float32, name="input"))
        print(f"Model input shape set to {input_shape}.")

    # Apply quantization if specified
    if quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        print("Applying int8 quantization for size optimization.")
    elif quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Applying float16 quantization for optimized speed and size.")
    elif quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying dynamic range quantization for general optimization.")
    else:
        print("No quantization applied. Using the full precision model.")

    # Convert the model
    try:
        tflite_model = converter.convert()
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model successfully converted and saved to '{output_path}'.")
    except Exception as e:
        print(f"Error during conversion: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert a Keras model to TensorFlow Lite format with optional quantization.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the Keras model file (.h5)")
    parser.add_argument('--output_path', type=str, default="converted_model.tflite", help="Path to save the converted TFLite model")
    parser.add_argument('--quantization', type=str, choices=['int8', 'float16', 'dynamic', None], default=None, help="Type of quantization to apply (int8, float16, dynamic)")
    parser.add_argument('--input_shape', type=int, nargs='+', help="Specify the input shape for the model (excluding batch size)")
    args = parser.parse_args()

    # Load the Keras model
    model = load_model(args.model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Convert and save the model to TFLite format
    convert_model_to_tflite(model, args.output_path, args.quantization, args.input_shape)

if __name__ == "__main__":
    main()
