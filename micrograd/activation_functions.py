from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Value


class Activation:
    @staticmethod
    def relu(x: "Value") -> "Value":
        from .engine import Value

        return x if x.data > 0 else Value(0)

    @staticmethod
    def linear(x: "Value") -> "Value":
        return x

    @staticmethod
    def sigmoid(x: "Value") -> "Value":
        return 1 / (1 + (-x).exp())

    @staticmethod
    def tanh(x: "Value") -> "Value":
        return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
