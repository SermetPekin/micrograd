import random
from typing import List, Callable, Optional
from abc import ABC, abstractmethod

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
                Activation.relu
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

    def clone(self):
        import copy
        return copy.deepcopy(self)


class OptimizerAbstract(ABC):
    """OptimizerAbstract"""


class Optimizer(OptimizerAbstract):
    @staticmethod
    def step(parameters: List[Value], learning_rate: float) -> None:
        for param in parameters:
            param.data -= learning_rate * param.grad


class OptimizerForComparison(OptimizerAbstract):
    def __init__(self, parameters: List[Value] = (), learning_rate: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}  # {id(param): 0.0 for param in parameters}  # Momentum storage

    def __call__(self, *args, **kw) -> 'OptimizerForComparison':
        return self

    def step(self, parameters=(), learning_rate: float = None) -> None:
        if parameters:
            self.parameters = parameters

        if learning_rate:
            self.learning_rate = learning_rate

        for param in self.parameters:

            if id(param) not in self.velocities:
                self.velocities[id(param)] = 0.0

            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                param.grad += self.weight_decay * param.data

            # Apply momentum
            velocity = self.momentum * self.velocities[id(param)] - self.learning_rate * param.grad
            self.velocities[id(param)] = velocity

            # Update parameters
            param.data += velocity


class Trainer:
    def __init__(
            self,
            model: Module,
            loss_fn: Callable[[Value, Value], Value],
            optimizer: OptimizerAbstract,
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


class TrainerForComparison:
    def __init__(
            self,
            model: Module,
            loss_fn: Callable[[Value, Value], Value],
            optimizer: OptimizerAbstract,
            num_clones: int = 1,
            eval_interval: int = 200,
    ):
        self.models = [model.clone() for _ in range(num_clones)]
        self.loss_fn = loss_fn
        self.optimizers = [optimizer(model.parameters()) for model in
                           self.models]  # [optimizer(model.parameters()) for model in self.models]
        self.eval_interval = eval_interval
        self.best_model = None

    def train(
            self,
            inputs: List[List[Value]],
            targets: List[Value],
            epochs: int,
            learning_rate: float,
    ) -> None:
        for epoch in range(epochs):
            if epoch < self.eval_interval:
                # Train all clones during the evaluation interval
                for index, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                    model.number = index + 1
                    self._train_one_epoch(index, model, optimizer, inputs, targets, learning_rate)
            elif epoch == self.eval_interval:
                # Evaluate clones and select the best one
                self.best_model = self._evaluate_and_select_best(inputs, targets)
                print(f"After {self.eval_interval} epochs, best model selected.")
            else:
                # Train only the best model
                self._train_one_epoch( self.best_model.number, self.best_model, self.optimizers[0], inputs, targets, learning_rate)

    def _train_one_epoch(
            self,
            index: int,
            model: Module,
            optimizer: OptimizerAbstract,
            inputs: List[List[Value]],
            targets: List[Value],
            learning_rate: float,
    ) -> None:
        total_loss = 0
        for input_data, target in zip(inputs, targets):
            # Forward pass
            predictions = model(input_data)
            loss = self.loss_fn(predictions, target)
            total_loss += loss.data

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step(model.parameters(), learning_rate)

            # Zero gradients for the next iteration
            model.zero_grad()

        print(f"Training Loss: {total_loss / len(inputs):.4f}")

        # print(f"Epoch {epoch + 1}/{epochs}, Clone {index + 1}, Training Loss: {total_loss / len(inputs):.4f}")
        print(f"Epoch  Clone {index + 1}, Training Loss: {total_loss / len(inputs):.4f}")
        # print(f"Clone {i + 1}, Validation Loss: {avg_loss:.4f}")

    def _evaluate_and_select_best(self, inputs: List[List[Value]], targets: List[Value]) -> Module:
        best_loss = float("inf")
        best_model = None
        for model in self.models:
            total_loss = 0
            for input_data, target in zip(inputs, targets):
                predictions = model(input_data)
                loss = self.loss_fn(predictions, target)
                total_loss += loss.data
            avg_loss = total_loss / len(inputs)
            print(f"Model Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model
        return best_model
