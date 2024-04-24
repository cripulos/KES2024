import tensorflow as tf
# from tensorflow import keras

converter = tf.lite.TFLiteConverter.from_saved_model('../Model/breast/prunned_TF_lite')
# converter.summary()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

quantized_model_path = '../Model/quantized_model_2.tflite'
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Quantized model saved to: {quantized_model_path}")


interpreter = tf.lite.Interpreter(model_path='../Model/quantized_model_2.tflite')

# Allocate memory for the interpreter
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Display input details
for detail in input_details:
    print(detail)

# Display output details
for detail in output_details:
    print(detail)