import tensorflow as tf

model = tf.keras.models.load_model("models/gru_gesture_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable TF Select ops (lets TFLite run GRU/LSTM ops that are not natively supported)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # default TFLite ops
]

# Disable experimental lowering of tensor list ops
converter.experimental_enable_resource_variables  = False
converter._experimental_lower_tensor_list_ops = True

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("models/gru_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ GRU model exported to TFLite successfully!")
