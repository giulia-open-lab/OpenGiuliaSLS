import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

print("Program Started\nComparison between big model using torch, CPU and GPU\n")

# Neural network definition
class LargeNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LargeNeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def summary(self):
        print("\nModel Summary:\n")
        total_params = 0
        input_size = (1, 1024)  # Assuming the input is a single sample with size 1024
        x = torch.zeros(input_size)
        print(f"{'Layer':<20} {'Input Shape':<30} {'Output Shape':<30} {'Param #':<15}")
        print("="*95)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                out = layer(x)
                num_params = layer.in_features * layer.out_features + layer.out_features
                print(f"{layer.__class__.__name__:<20} {str(tuple(x.shape)):<30} {str(tuple(out.shape)):<30} {num_params:<15}")
                total_params += num_params
                x = out
            elif isinstance(layer, nn.ReLU):
                out = layer(x)
                print(f"{layer.__class__.__name__:<20} {str(tuple(x.shape)):<30} {str(tuple(out.shape)):<30} {'-':<15}")
                x = out
        print("="*95)
        print(f"Total Parameters: {total_params}\n")

# Function to measure training time
def train_model(device, epochs=5, input_size=1024, hidden_size=512, output_size=10, num_layers=10, batch_size=256):
    # Generate random data
    X = torch.randn(10000, input_size).to(device)
    y = torch.randint(0, output_size, (10000,)).to(device)
    
    # Define model, loss, and optimizer
    model = LargeNeuralNetwork(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for i in range(0, X.size(0), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Zero the gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Optimization
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    end_time = time.time()
    print(f"Training time on {device}: {end_time - start_time:.2f} seconds\n")
    return model, end_time - start_time

# Training on CPU
model_CPU, elapsed_CPU = train_model(torch.device("cpu"))

# Training on GPU (if available)
if torch.cuda.is_available():
    model_GPU, elapsed_GPU = train_model(torch.device("cuda"))
    print(f"CPU time = {elapsed_CPU:.2f}")
    print(f"GPU time = {elapsed_GPU:.2f}")
    print(f"GPU time / CPU time = {elapsed_GPU/elapsed_CPU:.2f}")
    # Print model summary once at the end
    print("\nFinal Model Summary:")
    model_CPU.summary()
    sys.exit(1)
else:
    print("GPU not available. Run the code on a machine with a GPU to see the benefits.")

    # Print model summary once at the end
    print("\nFinal Model Summary:")
    model_CPU.summary()
    sys.exit(0)
