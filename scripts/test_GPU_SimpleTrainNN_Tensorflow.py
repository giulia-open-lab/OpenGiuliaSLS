import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import time
import sys
start_time = time.time()

print("\nTEST GPU SIMPLE TRAIN NN - Tensorflow")
print("program start...")

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    device = "/GPU:0"
    print("\nUsing device: GPU\n")
else:
    device = "/CPU:0"
    print("\nUsing device: CPU\n")

# Define a simple neural network with two layers, each having 4 nodes
class SimpleNet(tf.keras.Model):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = tf.keras.layers.Dense(4, activation='relu')  # Input size 4, output size 4
        self.layer2 = tf.keras.layers.Dense(4)  # Input size 4, output size 4

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNet()

# Define a loss function and optimizer
criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Dummy data for training (4 features per sample)
x_train = tf.constant([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 1.0, 1.1, 1.2],
                       [1.3, 1.4, 1.5, 1.6]], dtype=tf.float32)
y_train = tf.constant([[0.2, 0.4, 0.6, 0.8],
                       [1.0, 1.2, 1.4, 1.6],
                       [1.8, 2.0, 2.2, 2.4],
                       [2.6, 2.8, 3.0, 3.2]], dtype=tf.float32)

# Training loop
num_epochs = 100

# Use tf.device to ensure GPU/CPU usage
with tf.device(device):
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(x_train)
            loss = criterion(y_train, outputs)

        # Backward pass and optimization
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'{device} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')

print(f"\nProgram Terminated.\n{device} Total Time {time.time()-start_time:0.4f}")

if device=="/GPU:0": 
    sys.exit(1)
else: 
    sys.exit(0)