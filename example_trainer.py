from micrograd import Value, MLP, Optimizer , Trainer

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


# Model
model = MLP(input_size=2, layer_sizes=[3, 1])

# Optimizer
optimizer = Optimizer()

# Trainer
trainer = Trainer(model=model, loss_fn=mean_squared_error, optimizer=optimizer)

# Train
trainer.train(inputs, targets, epochs=100, learning_rate=0.01)

# Test
test_input = [Value(5.0), Value(6.0)]  # Expected output: 31
prediction = model(test_input)
print(f"Prediction for input {test_input}: {prediction.data:.4f}")
