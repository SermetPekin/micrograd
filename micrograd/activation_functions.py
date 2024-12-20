from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Value


class Activation:
    @staticmethod
    def relu(value: "Value") -> "Value":
        return value.relu()

    @staticmethod
    def linear(value: "Value") -> "Value":
        return value

    @staticmethod
    def sigmoid(value: "Value") -> "Value":
        return value.sigmoid()

    @staticmethod
    def tanh(value: "Value") -> "Value":
        return value.tanh()
