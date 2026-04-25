import sys
import time

import tensorflow as tf

input_dim = 100
num_samples = 50000

print("Program Started\nComparison between big model using tensorflow, CPU and GPU\n")
# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus}")
else:
    print("GPU not available, training will be performed on the CPU.")

# Function to create a simple random dataset
def create_dataset():
    x = tf.random.normal((num_samples, input_dim))
    y = tf.random.uniform((num_samples, 1), maxval=2, dtype=tf.int32)
    return x, y

# Create the training dataset
X_train, y_train = create_dataset()

# Define a large neural network
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model and measure the time taken
def train_model_on_device(x, y, epochs=5, batch_size=256, device='/CPU:0'):
    with tf.device(device):
        model = create_model()
        start_time = time.time()
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
        end_time = time.time()
        print(f"Training time on {device}: {end_time - start_time:.2f} seconds")
    return model, end_time - start_time

# Compare CPU and GPU training
print("Training on CPU...")
model_cpu, elapsed_CPU = train_model_on_device(X_train, y_train, device='/CPU:0')

if gpus:
    print("Training on GPU...")
    model_gpu, elapsed_GPU = train_model_on_device(X_train, y_train, device='/GPU:0')
    print(f"\nCPU time = {elapsed_CPU:.2f}")
    print(f"GPU time = {elapsed_GPU:.2f}")
    print(f"GPU time / CPU time = {elapsed_GPU/elapsed_CPU:.2f}\n")


# NOTE: To test the effectiveness of the GPU compared to the CPU, 
# you can force execution on the CPU using "with tf.device('/CPU:0')".
# This way, you can compare the execution times of both versions.

# Summary of the model structure
print("\nModel Summary:")
model_cpu.summary()

if gpus: 
    sys.exit(1)
else: 
    sys.exit(0)