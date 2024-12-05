import pytest
import torch
from micrograd import Value, MLP, Optimizer, Trainer, OptimizerForComparison, TrainerForComparison


class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Loss function for micrograd
def mean_squared_error(predicted: Value, target: Value) -> Value:
    return (predicted - target) ** 2


def initialize_weights_micrograd(model):
    for layer in model.layers:
        for neuron in layer.neurons:
            for weight in neuron.weights:
                weight.data = 0.5  # Example fixed value
            neuron.bias.data = 0.1  # Example fixed value


def initialize_weights_torch(model):
    with torch.no_grad():
        model.fc1.weight.fill_(0.5)
        model.fc1.bias.fill_(0.1)
        model.fc2.weight.fill_(0.5)
        model.fc2.bias.fill_(0.1)


def data1():
    inputs = [
        [Value(1.0), Value(2.0)],
        [Value(2.0), Value(3.0)],
        [Value(3.0), Value(4.0)],
        [Value(4.0), Value(5.0)]
    ]
    targets = [Value(9.0), Value(14.0), Value(19.0), Value(24.0)]

    torch_inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    torch_targets = torch.tensor([[9.0], [14.0], [19.0], [24.0]])
    return inputs, targets, torch_inputs, torch_targets


def data2():
    inputs = [[Value(i), Value(i + 1)] for i in range(1, 21)]
    targets = [Value(2 * i + 3 * (i + 1) + 1) for i in range(1, 21)]
    torch_inputs = torch.tensor([[i, i + 1] for i in range(1, 21)], dtype=torch.float32)
    torch_targets = torch.tensor([[2 * i + 3 * (i + 1) + 1] for i in range(1, 21)], dtype=torch.float32)
    return inputs, targets, torch_inputs, torch_targets


# @pytest.mark.skipif(True, reason='TODO')
def compare_micrograd_vs_torch():
    # Dataset
    inputs, targets, torch_inputs, torch_targets = data1()

    # Micrograd Model
    micrograd_model = MLP(input_size=2, layer_sizes=[3, 1])
    micrograd_optimizer = OptimizerForComparison()
    micrograd_trainer = TrainerForComparison(
        model=micrograd_model,
        loss_fn=mean_squared_error,
        optimizer=micrograd_optimizer,
        num_clones=5
    )

    # initialize_weights_micrograd(micrograd_model)

    EPOCHS = int(10000)
    # Train Micrograd Model
    micrograd_trainer.train(inputs, targets, epochs=EPOCHS, learning_rate=0.01)

    # PyTorch Model
    torch_model = TorchMLP()
    # initialize_weights_torch(torch_model)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Train PyTorch Model
    for epoch in range(EPOCHS):
        torch_optimizer.zero_grad()
        predictions = torch_model(torch_inputs)
        loss = loss_fn(predictions, torch_targets)
        loss.backward()
        torch_optimizer.step()

    # Compare Predictions
    micrograd_test_input = [Value(5.0), Value(6.0)]
    micrograd_prediction = micrograd_model(micrograd_test_input).data

    torch_test_input = torch.tensor([[5.0, 6.0]])
    torch_prediction = torch_model(torch_test_input).item()

    msg = f'micrograd_prediction: {micrograd_prediction} torch_prediction :  {torch_prediction}'
    print(msg)
    # Assert that predictions are close
    # assert pytest.approx(micrograd_prediction,
    #                      rel=1e-2) == torch_prediction, f'micrograd_prediction: {micrograd_prediction} torch_prediction :  {torch_prediction}'


compare_micrograd_vs_torch()
