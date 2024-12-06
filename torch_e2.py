import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def init_data():
    def fnc(d: str):
        dict_ = {
            'Setosa': 0,
            'Versicolor': 1,
            'Virginica': 2,

        }
        return dict_.get(d, d)

    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    df = pd.read_csv(url)
    df['variety'] = df['variety'].apply(fnc)
    return df


def process_data_for_torch(df):
    X = df.drop('variety', axis=1)
    y = df['variety']
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return X_train, X_test, y_train, y_test


class Model(nn.Module):
    def __init__(self, in_feats: int = 4, out_feats: int = 3, hidden1=7, hidden2=7):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_feats)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def with_torch(save=True):
    df = init_data()
    X_train, X_test, y_train, y_test = process_data_for_torch(df)

    model = Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.parameters
    epochs = 60
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)

        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'Epoch : {i} and loss : {loss}')

        # backprop

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if save:
        # Save the model
        torch.save(model.state_dict(), "iris_model.pth")

    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test).item()
        accuracy = calculate_accuracy(y_eval, y_test)

        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return model, losses


def load_model():
    # Load the model
    loaded_model = Model()
    loaded_model.load_state_dict(torch.load("iris_model.pth"))
    loaded_model.eval()
    return loaded_model


def calculate_accuracy(_y_pred, _y_true):
    y_pred_classes = torch.argmax(_y_pred, axis=1)  # Get the predicted class
    acc = (y_pred_classes == _y_true).sum().item() / len(_y_true)
    return acc


# with_torch()

def with_micrograd():
    from micrograd import Value
    from micrograd import MLP

    # Define the Micrograd model
    in_feats = 4  # Input features (Iris dataset)
    hidden1 = 7  # Hidden layer 1
    hidden2 = 7  # Hidden layer 2
    out_feats = 3  # Output classes

    model = MLP(in_feats, [hidden1, hidden2, out_feats])  # Equivalent to your PyTorch model
    df = init_data()
    X_train, X_test, y_train, y_test = process_data_for_torch(df)
    # Hyperparameters
    learning_rate = 0.01
    epochs = 100
    losses = []

    # # # One-hot encode y_train
    # y_train_onehot = np.zeros((y_train.size, out_feats))
    # y_train_onehot[np.arange(y_train.size), y_train] = 1
    # One-hot encode y_train
    y_train_onehot = np.zeros((y_train.shape[0], out_feats))  # Initialize with zeros
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1  # Set the appropriate index to 1


    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0

        for i in range(len(X_train)):
            # Forward pass
            inputs = [Value(x) for x in X_train[i]]
            targets = [Value(y) for y in y_train_onehot[i]]
            outputs = model(inputs)

            # Calculate Cross-Entropy Loss
            exp_outputs = [o.exp() for o in outputs]
            sum_exp_outputs = sum(exp_outputs)
            probs = [o / sum_exp_outputs for o in exp_outputs]
            loss = -sum(t * p.log() for t, p in zip(targets, probs))

            epoch_loss += loss.data

            # Backpropagation
            model.zero_grad()  # Zero gradients
            loss.backward()

            # Update weights
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        losses.append(epoch_loss / len(X_train))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(X_train)}")


    # Evaluation
    correct = 0
    total = len(X_test)

    for i in range(len(X_test)):
        inputs = [Value(x) for x in X_test[i]]
        outputs = model(inputs)
        predicted = np.argmax([o.data for o in outputs])
        if predicted == y_test[i]:
            correct += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # import matplotlib.pyplot as plt

    # Plot the losses
    plt.plot(range(epochs), losses, label='Micrograd Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Micrograd Loss Curve')
    plt.legend()
    plt.show()


with_micrograd()
