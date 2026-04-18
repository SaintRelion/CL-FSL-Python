# python
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

# Matches the new Dual-Hand vector from your pillar_collector.py
TARGET_FRAMES = 30
FEATURE_COUNT = 50  # (Num_Shapes + 5 Wrist + 5 Tip + 1 Presence) * 2
INPUT_SHAPE = (TARGET_FRAMES, FEATURE_COUNT)

# ------------------- Data Loading -------------------
# Loading the X.npy and y.npy built by your updated collector
X = np.load("data/X.npy")
y = np.load("data/y.npy")

num_classes = len(np.unique(y))

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")
print(f"Number of classes: {num_classes}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ------------------- Model Architecture -------------------
# Using the hierarchical logic where we process Shape Probs + Pillar Paths
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=INPUT_SHAPE, name="jutsu_input"),
        tf.keras.layers.Masking(mask_value=0.0),  # Ignore padding/missing hand frames
        # First GRU layer - Unrolled for TFLite compatibility
        tf.keras.layers.GRU(
            64, return_sequences=True, unroll=True, reset_after=True, name="gru_path_1"
        ),
        tf.keras.layers.Dropout(0.3),
        # Second GRU layer - Finalizing the temporal sequence
        tf.keras.layers.GRU(64, unroll=True, reset_after=True, name="gru_path_2"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# ------------------- Training Configuration -------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Callbacks for AWS run
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
]

# ------------------- Execute Training -------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
)

# ------------------- Export to TFLite -------------------
os.makedirs("models", exist_ok=True)
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
