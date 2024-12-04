# Example: Training a simple MLP
import random
from typing import List, Union

from micrograd.engine import Value
from micrograd.nn import MLP
import time

SECONDS_TO_WAIT = 0  # 0.1

# Create random data
inputs = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(10)]
targets = [random.uniform(-1, 1) for _ in range(10)]

# Define the MLP
mlp: MLP = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers with 4 neurons each, 1 output

# Training loop
learning_rate = 0.01
if SECONDS_TO_WAIT > 0:
    print(f'it will wait {SECONDS_TO_WAIT:0.01f} seconds to imitate the real calculation ')
    time.sleep(2)

for epoch in range(100):

    # time.sleep(SECONDS_TO_WAIT)
    # Forward pass
    predictions: List[Value] = [mlp(x) for x in inputs]
    # Mean Squared Error
    loss: Value = sum((pred - target) ** 2 for pred, target in zip(predictions, targets))
    assert isinstance(loss, Value) , 'it is not Value '

    # Zero gradients
    for p in mlp.parameters():
        p.grad = 0

    # Backward pass
    loss.backward()

    # Gradient descent step
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad

    print(f"Epoch {epoch}, Loss: {loss.data:0.03f}")
