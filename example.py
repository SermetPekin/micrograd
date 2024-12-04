from micrograd import Value, draw_dot


def example_1():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    print(f'{g.data:.4f}')  # prints 24.7041, the outcome of this forward pass
    g.backward()
    print(f'{a.grad:.4f}')  # prints 138.8338, i.e. the numerical value of dg/da
    print(f'{b.grad:.4f}')  # prints 645.5773, i.e. the numerical value of dg/db


def example_2():
    from micrograd import nn, Value
    n = nn.Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    # draw_dot function was defined in `trace_graph.ipynb` file.
    dot = draw_dot(y)
    assert dot


if __name__ == '__main__':
    example_1()
    example_2()
