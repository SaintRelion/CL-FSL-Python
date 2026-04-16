# python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    print(f"✅ GPU Detected: {physical_devices}")
    # This prevents TF from hogging all VRAM immediately
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️ No GPU found. Running on CPU.")

# ------------------- Hyperparameters -------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 1  # 80  # Increased slightly since the features are now cleaner
LEARNING_RATE = 1e-3

# Represents the fixed sequence length (time steps)
TARGET_FRAMES = 30
# Hand1(7) + Hand2(7) + 10 Pillar Distances = 24
FEATURE_COUNT = 24
INPUT_SHAPE = (TARGET_FRAMES, FEATURE_COUNT)

# ------------------- Data Loading -------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

num_classes = len(np.unique(y))

print(f"Loaded X shape: {X.shape}")  # Should be (N, 30, 18)
print(f"Loaded y shape: {y.shape}")
print(f"Number of classes: {num_classes}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,  # stratify=y
)

# ------------------- Model Architecture -------------------
# We use Bidirectional to capture the "preparation" and "retraction" of signs
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_layer"),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                64,
                return_sequences=True,
                unroll=True,  # <--- CRITICAL: This flattens the loop for TFLite
                reset_after=True,
            )
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64, unroll=True, reset_after=True),  # <--- CRITICAL
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

# Callbacks to prevent overfitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
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

# ------------------- Export -------------------
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.keras")

# Optional: Convert to TFLite for your Flutter app immediately
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, TARGET_FRAMES, FEATURE_COUNT], model.inputs[0].dtype)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Optimization settings
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,  # Required for some complex GRU ops
]

# This flag helps with the TensorList errors in some TF versions
converter._experimental_lower_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open("models/gesture_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("TFLite conversion successful!")
except Exception as e:
    print(f"Conversion failed: {e}")
