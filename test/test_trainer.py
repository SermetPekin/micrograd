from micrograd import Activation, Value

from micrograd import Value, MLP, Optimizer , Trainer , OptimizerForComparison

# Dataset
inputs = [
    [Value(1.0), Value(2.0)],
    [Value(2.0), Value(3.0)],
    [Value(3.0), Value(4.0)],
    [Value(4.0), Value(5.0)]
]
targets = [
    Value(9.0),
    Value(14.0),
    Value(19.0),
    Value(24.0)
]


# Loss function
def mean_squared_error(predicted: Value, target: Value) -> Value:
    return (predicted - target) ** 2



import torch
from micrograd.engine import Value


def test_sanity_check_with_trainer():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_complete_train(capsys):
    with capsys.disabled():

        # Model
        model = MLP(input_size=2, layer_sizes=[3, 1])

        # Optimizer
        optimizer = OptimizerForComparison()

        # Trainer
        trainer = Trainer(model=model, loss_fn=mean_squared_error, optimizer=optimizer)

        # Train
        trainer.train(inputs, targets, epochs=100, learning_rate=0.01)

        # Test
        test_input = [Value(5.0), Value(6.0)]  # Expected output: 31
        prediction = model(test_input)
        print(f"Prediction for input {test_input}: {prediction.data:.4f}")



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
