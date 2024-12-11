# Example: Training a simple MLP
import random
from typing import List

from micrograd.engine import Value
from micrograd.nn import MLP
import time
from micrograd.adam import Adam

SECONDS_TO_WAIT = 0  # 0.1

# Create random data
inputs = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(10)]
targets = [random.uniform(-1, 1) for _ in range(10)]

# Define the MLP
mlp: MLP = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers with 4 neurons each, 1 output
optimizer = Adam(mlp.parameters(), lr=0.01)


# Training loop
learning_rate = 0.01

for epoch in range(100):

    # Forward pass
    predictions: List[Value] = [mlp(x) for x in inputs]
    # Mean Squared Error
    loss: Value = sum((pred - target) ** 2 for pred, target in zip(predictions, targets))
    assert isinstance(loss, Value) , 'it is not Value '

    # Zero gradients
    # for p in mlp.parameters():
    #     p.grad = 0
    optimizer.zero_grad() # Adam optimizer instead of SGD
    # Backward pass
    loss.backward()

    optimizer.step()

    # Gradient descent step
    # for p in mlp.parameters():
    #     p.data -= learning_rate * p.grad

    print(f"Epoch {epoch}, Loss: {loss.data:0.03f}")
