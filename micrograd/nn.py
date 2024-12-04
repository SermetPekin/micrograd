import random
from typing import List, Union

from micrograd.engine import Value, Weight, Bias


class Module:

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):

    def __init__(self, input_count: int, use_activation: bool = True):
        self.weights: List[Weight] = [Weight(random.uniform(-1, 1)) for _ in range(input_count)]
        self.bias: Bias = Bias(0)
        self.use_activation: bool = use_activation

    def __call__(self, inputs: List[Value]) -> Value:
        weighted_sum = sum((weight * input_value for weight, input_value in zip(self.weights, inputs)), self.bias)
        return weighted_sum.relu() if self.use_activation else weighted_sum

    def parameters(self) -> List[Value]:
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        activation_type = "ReLU" if self.use_activation else "Linear"
        return f"{activation_type}Neuron(input_count={len(self.weights)})"

class Layer(Module):

    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons: List[Neuron] = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x) -> List[Value] | Value:
        out: List[Value] = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, input_size: int, layer_sizes: List[int]):
        sizes: List[int] = [input_size] + layer_sizes
        self.layers: List[Layer] = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(layer_sizes) - 1) for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs: List[Value]) -> Union[Value, List[Value]]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def parameters(self) -> List[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
