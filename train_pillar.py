import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# ------------------- GPU Configuration -------------------
# This ensures TensorFlow uses the AWS L4 GPU efficiently without hogging all VRAM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(e)

# ------------------- Hyperparameters -------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3

TARGET_FRAMES = 30
FEATURE_COUNT = 30  # (7 Wrist + 7 Tip + 1 Presence) * 2 hands
INPUT_SHAPE = (TARGET_FRAMES, FEATURE_COUNT)

X = np.load("data/X.npy")
y = np.load("data/y.npy")
num_classes = len(np.unique(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(30, FEATURE_COUNT)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.GRU(64, return_sequences=True, unroll=True),
        tf.keras.layers.GRU(64, unroll=True),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

model.save("models/pillar_path_model.keras")

# Using the concrete function approach for maximum TFLite stability
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, TARGET_FRAMES, FEATURE_COUNT], model.inputs[0].dtype)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open("models/pillar_path_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("TFLite conversion successful!")
except Exception as e:
    print(f"Conversion failed: {e}")
