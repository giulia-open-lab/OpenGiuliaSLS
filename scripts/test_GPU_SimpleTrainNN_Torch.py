import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

start_time = time.time()

print("\nTEST GPU SIMPLE TRAIN NN - Torch")
print("program start...")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# Define a simple neural network with two layers, each having 4 nodes
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(4, 4)  # Input size 4, output size 4
        self.layer2 = nn.Linear(4, 4)  # Input size 4, output size 4

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Instantiate the model and move it to the GPU if available
model = SimpleNet().to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data for training (4 features per sample)
x_train = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                    [1.3, 1.4, 1.5, 1.6]], dtype=torch.float32).to(device)
y_train = torch.tensor([[0.2, 0.4, 0.6, 0.8],
                    [1.0, 1.2, 1.4, 1.6],
                    [1.8, 2.0, 2.2, 2.4],
                    [2.6, 2.8, 3.0, 3.2]], dtype=torch.float32).to(device)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'{device} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        #time.sleep(1)

print(f"\nProgram Terminated.\n{device} Total Time {time.time()-start_time:0.4f}")

if device.type=="cuda": 
    sys.exit(1)
else: 
    sys.exit(0)