from sklearn.datasets import load_iris
import pandas as pd
from typing import Tuple

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

from micrograd.data import iris_data
from micrograd.adam import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def process_data_for_torch_and_micrograd(df: pd.DataFrame):
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


def get_iris_data_split() -> Tuple['X_train', 'X_test', 'y_train', 'y_test']:
    df = iris_data()
    return process_data_for_torch_and_micrograd(df)


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


from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    precision = precision_score(y_true, predicted_classes, average='weighted')
    recall = recall_score(y_true, predicted_classes, average='weighted')
    f1 = f1_score(y_true, predicted_classes, average='weighted')

    return precision, recall, f1


def calc_torch_roc(model, X_test, y_test):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_logits = model(X_test)  # Get raw logits
        y_probs = torch.softmax(y_logits, dim=1)  # Convert logits to probabilities

    # Convert predictions and targets to numpy arrays
    y_probs_np = y_probs.numpy()
    y_test_np = y_test.numpy()

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(
        y_test_np,
        y_probs_np,
        multi_class='ovr',  # 'ovr' for one-vs-rest (multi-class)
        average='weighted'  # Weighted average
    )

    print(f"ROC AUC Score: {roc_auc:.4f}")


def with_torch(save=True):
    df = iris_data()
    X_train, X_test, y_train, y_test = process_data_for_torch_and_micrograd(df)

    model = Model()

    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.parameters
    epochs = 100
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = cross_entropy(y_pred, y_train)

        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'Epoch : {i} and loss : {loss}')

        # backprop

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if save:
        # Save the model
        torch.save(model.state_dict(), "iris_model2.pth")

    model.eval()
    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = cross_entropy(y_eval, y_test).item()
        accuracy = calculate_accuracy(y_eval, y_test)

        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        # return model, losses

        precision, recall, f1 = calculate_metrics(y_eval, y_test)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        calc_torch_roc(model, X_test, y_test)

    # Plot the losses
    plt.plot(range(epochs), losses, label='Losses with pytorch (python)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Torch Loss Curve')
    plt.legend()
    plt.show()


def with_micrograd():
    from micrograd import Value
    from micrograd import MLP

    # Define the Micrograd model
    in_feats = 4  # Input features (Iris dataset)
    hidden1 = 7  # Hidden layer 1
    hidden2 = 7  # Hidden layer 2
    out_feats = 3  # Output classes

    model = MLP(in_feats, [hidden1, hidden2, out_feats])
    optimizer = Adam(model.parameters(), lr=0.01)

    X_train, X_test, y_train, y_test = get_iris_data_split
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
            # model.zero_grad()  # Zero gradients
            optimizer.zero_grad()  # Zero gradients

            loss.backward()
            # optimizer.step()

            optimizer.step()
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
    plt.plot(range(epochs), losses, label='MicrogradExtended Loss [python]')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Micrograd Loss Curve')
    plt.legend()
    plt.show()


with_torch()

# with_micrograd()
