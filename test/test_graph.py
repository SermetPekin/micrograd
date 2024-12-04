from micrograd import nn, Value, draw_dot


def test_graph():
    n = nn.Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    dot = draw_dot(y)
    assert dot
