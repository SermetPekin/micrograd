import math
from typing import Callable, Set, Union

Number = Union[int, float]


class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data: Number | 'Value', _children: tuple = (), _op: str = ""):
        if isinstance(data, Value):
            data.copy(data, _children, _op)
        else:
            self.data: Number = data
            self.grad: float = 0.0
            # internal variables used for autograd graph construction
            self._backward: Callable[[], None] = lambda: None
            self._prev: Set["Value"] = set(_children)
            self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def copy(self, value: 'Value', _children: tuple = (), _op: str = ""):
        self.data = value.data
        self.grad = value.grad
        self._backward = value._backward
        self._prev = value._prev
        self._op = value._op
        if _children:
            self._prev: Set["Value"] = set(_children)
        if _op:
            self._op = _op

    def __add__(self, other: Number | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Number | "Value"):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: Number) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:

        # topological order all the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> "Value":  # -self
        return self * -1

    def __radd__(self, other) -> "Value":  # other + self
        return self + other

    def __sub__(self, other) -> "Value":  # self - other
        return self + (-other)

    def __rsub__(self, other) -> "Value":  # other - self
        return other + (-self)

    def __rmul__(self, other) -> "Value":  # other * self
        return self * other

    def __truediv__(self, other) -> "Value":  # self / other
        return self * other ** -1

    def __rtruediv__(self, other) -> "Value":  # other / self
        return other * self ** -1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out


class Weight(Value):
    def __init__(
            self,
            data: Number,
            _children: tuple = (),
            _op: str = "",
            regularization: str = "none",
    ):
        super().__init__(data, _children, _op)
        self.regularization = regularization


class Bias(Value):
    def __init__(self, data: Number, _children: tuple = (), _op: str = ""):
        super().__init__(data, _children, _op)
