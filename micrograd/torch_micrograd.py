import torch
import torch.nn as nn
import torch.nn.functional as F


class Value:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    def __repr__(self):
        return f"Value(data={self.data.item()}, grad={self.data.grad})"


class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        layers = []
        sizes = [input_size] + layer_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation for all but the last layer
                x = F.relu(x)
        return x


class Optimizer:
    def __init__(self, parameters, learning_rate=0.01):
        self.optimizer = torch.optim.SGD(parameters, lr=learning_rate)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def train(model, inputs, targets, loss_fn, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0.0

        # Forward pass
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    inputs_ = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    targets_ = torch.tensor([[9.0], [14.0], [19.0], [24.0]])

    model_ = MLP(input_size=2, layer_sizes=[3, 1])
    loss_fn_ = nn.MSELoss()
    optimizer_ = Optimizer(model_.parameters(), learning_rate=0.01)

    train(model_, inputs_, targets_, loss_fn_, optimizer_, epochs=100)

    test_input = torch.tensor([[5.0, 6.0]])
    prediction = model_(test_input)
    print(f"Prediction for input {test_input.tolist()}: {prediction.tolist()}")
