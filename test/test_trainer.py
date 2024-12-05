from micrograd import Activation, Value


def test_relu():
    assert Activation.relu(Value(3.0)).data == Value(3.0).data
    assert Activation.relu(Value(-2.0)).data == Value(0.0).data


def test_linear():
    assert Activation.linear(Value(3.0)).data == Value(3.0).data
    assert Activation.linear(Value(-2.0)).data == Value(-2.0).data


def test_sigmoid():
    sigmoid_value = Activation.sigmoid(Value(0.0))
    assert isinstance(sigmoid_value, Value)
    assert abs(sigmoid_value - 0.5) < 1e-6


def test_tanh():
    tanh_value = Activation.tanh(Value(0.0))
    assert isinstance(tanh_value, Value)
    assert abs(tanh_value) < 1e-6
