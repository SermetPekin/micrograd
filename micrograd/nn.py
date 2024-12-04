import random
from typing import List, Callable, Optional

from micrograd.engine import Value, Weight, Bias


class Module:

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):

    def __init__(
            self,
            input_count: int,
            activation_function: Optional[Callable[[Value], Value]] = None,
    ):
        from .activation_functions import Activation

        if activation_function is None:
            self.activation_function: Optional[Callable[[Value], Value]] = (
                Activation.linear
            )
        else:
            self.activation_function: Optional[Callable[[Value], Value]] = (
                activation_function
            )

        self.weights: List[Weight] = [
            Weight(random.uniform(-1, 1)) for _ in range(input_count)
        ]
        self.bias: Bias = Bias(0)

    def __call__(self, inputs: List[Value]) -> Value:
        weighted_sum: Value = sum(
            (weight * input_value for weight, input_value in zip(self.weights, inputs)),
            self.bias,
        )
        if self.activation_function is not None:
            return self.activation_function(weighted_sum)
        return weighted_sum

    def __repr__(self) -> str:
        activation_name = (
            self.activation_function.__name__
            if hasattr(self.activation_function, "__name__")
            else "Custom"
        )
        return f"{activation_name}Neuron(input_count={len(self.weights)})"

    def parameters(self) -> List[Value]:
        return self.weights + [self.bias]


class Layer(Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation_function: Callable[[Value], Value] | None = None,
    ):
        self.neurons: List[Neuron] = [
            Neuron(input_size, activation_function=activation_function)
            for _ in range(output_size)
        ]

    def __call__(self, inputs: List[Value]) -> List[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> List[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"


class MLP(Module):

    def __init__(self, input_size: int, layer_sizes: List[int]):
        from .activation_functions import Activation

        sizes: List[int] = [input_size] + layer_sizes
        self.layers: List[Layer] = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation_function=(
                    Activation.relu if i != len(layer_sizes) - 1 else Activation.linear
                ),
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs: List[Value]) -> List[Value]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> List[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class Optimizer:
    @staticmethod
    def step(parameters: List[Value], learning_rate: float) -> None:
        for param in parameters:
            param.data -= learning_rate * param.grad


class Trainer:
    def __init__(
            self,
            model: Module,
            loss_fn: Callable[[Value, Value], Value],
            optimizer: Optimizer,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(
            self,
            inputs: List[List[Value]],
            targets: List[Value],
            epochs: int,
            learning_rate: float,
    ) -> None:
        for epoch in range(epochs):
            total_loss = 0
            for input_data, target in zip(inputs, targets):
                # Forward pass
                predictions = self.model(input_data)
                loss = self.loss_fn(predictions, target)
                total_loss += loss.data

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step(self.model.parameters(), learning_rate)

                # Zero gradients for the next iteration
                self.model.zero_grad()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(inputs):.4f}")
