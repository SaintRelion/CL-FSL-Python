import numpy as np
import os

import tensorflow as tf
import pandas as pd


# ------------------- Robust Setup -------------------
DATA_PATH = "data/hand_vocabulary.csv"
os.makedirs("models", exist_ok=True)

# 1. Load Data manually to avoid complex TF-Data initializations
df = pd.read_csv(DATA_PATH, header=None)
X = df.iloc[:, 1:].values.astype("float32")  # 51 features
y_labels = df.iloc[:, 0].values

# 2. Simple Label Mapping
unique_labels = np.unique(y_labels)
label_map = {label: i for i, label in enumerate(unique_labels)}
y = np.array([label_map[l] for l in y_labels])

# Save labels for your Flutter app
with open("models/hand_shape_labels.txt", "w") as f:
    for label in unique_labels:
        f.write(f"{label}\n")

# ------------------- The AVX-Safe Model -------------------
# We use a flat architecture similar to your 'unrolled' pillar logic
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(51,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(unique_labels), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 3. Train (This path is usually safer than real-time prediction)
model.fit(X, y, epochs=50, batch_size=16)

# ------------------- Successful Export Logic -------------------
# Use the exact Concrete Function approach from train_pillar.py
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 51], model.inputs[0].dtype)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()
with open("models/hand_shape_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Hand Shape TFLite exported successfully!")
