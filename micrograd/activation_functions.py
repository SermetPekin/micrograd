from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Value


class Activation:
    @staticmethod
    def relu(value: "Value") -> "Value":
        from .engine import Value
        return value.relu()

    @staticmethod
    def linear(value: "Value") -> "Value":
        return value

    @staticmethod
    def sigmoid(value: "Value") -> "Value":
        from .engine import Value
        self = value
        out = Value(1 / (1 + (-value).exp()), (self,), "Sigmoid")

        # Value(0 if self.data < 0 else self.data, (self,), "Sigmoid")
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

        # return 1 / (1 + (-value).exp())

    @staticmethod
    def tanh(value: "Value") -> "Value":
        return (value.exp() - (-value).exp()) / (value.exp() + (-value).exp())
