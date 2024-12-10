from micrograd import Value
from micrograd import MLP
import random
from micrograd.adam import Adam

# Simple training data (XOR example)
inputs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)]
]
targets = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

# Define the model
model = MLP(2, [4, 1])  # 2 input neurons, 1 hidden layer with 4 neurons, 1 output neuron

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    total_loss = Value(0.0)
    for x, y in zip(inputs, targets):
        pred = model(x)[0]  # Assume model returns a list of outputs
        loss = (pred - y).pow(2)
        total_loss += loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Update weights
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.data:.4f}')

# Test the model
for x in inputs:
    print(f'Input: {x[0].data}, {x[1].data} -> Prediction: {model(x)[0].data:.4f}')
