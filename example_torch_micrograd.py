import torch
import torch.nn as nn
# import torch.nn.functional as F

from micrograd.torch_micrograd import train, Optimizer, MLP

# Example Usage
if __name__ == "__main__":
    # Create a dataset
    inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    targets = torch.tensor([[9.0], [14.0], [19.0], [24.0]])

    # Define the model, loss, and optimizer
    model = MLP(input_size=2, layer_sizes=[3, 1])
    loss_fn = nn.MSELoss()
    optimizer = Optimizer(model.parameters(), learning_rate=0.01)

    # Train the model
    train(model, inputs, targets, loss_fn, optimizer, epochs=100)

    # Test the model
    test_input = torch.tensor([[5.0, 6.0]])
    prediction = model(test_input)
    print(f"Prediction for input {test_input.tolist()}: {prediction.tolist()}")
