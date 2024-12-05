from micrograd import Activation, Value


def test_relu():
    assert Activation.relu(Value(3.0)).data == 3.0
    assert Activation.relu(Value(-2.0)).data == 0.0


def test_linear():
    assert Activation.linear(Value(3.0)).data == 3.0
    assert Activation.linear(Value(-2.0)).data == -2.0

#
# def test_sigmoid():
#     sigmoid_value = Activation.sigmoid(Value(0.0)).data
#     assert abs(sigmoid_value - 0.5) < 1e-6
#
#
# def test_tanh():
#     tanh_value = Activation.tanh(Value(0.0)).data
#     assert abs(tanh_value - 0.0) < 1e-6
