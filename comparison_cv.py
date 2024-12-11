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

from example_adam import epoch
from micrograd.data import iris_data
from micrograd.adam import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


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


from abc import ABC, abstractmethod


class SPNeuralNetworkImplBase(ABC):
    data_fnc = get_iris_data_split

    def __init__(self):
        ...

    @abstractmethod
    def set_data_fnc(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_fnc()

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def eval(self):
        ...

    @abstractmethod
    def show(self):
        ...


class ModelPYTORCH(nn.Module):
    def __init__(self, in_feats: int = 4, out_feats: int = 3, hidden1=7, hidden2=7):
        super(ModelPYTORCH, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_feats)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def iris_data_xy():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    data = load_iris()
    X = data.data
    y = data.target

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


class PyTorchImpl(SPNeuralNetworkImplBase):
    def __init__(self):
        super().__init__()
        self.model = ModelPYTORCH()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, *args, **kwargs):
        self.train()
        self.eval()
        self.show()

    def train(self, epochs=100, lr=0.01, save=True):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # model.parameters
        self.epochs = epochs
        self.losses = []
        for i in range(self.epochs):
            y_pred = self.model.forward(self.X_train)
            loss = self.cross_entropy(y_pred, self.y_train)
            self.losses.append(loss.detach().numpy())
            if i % 10 == 0:
                print(f'Epoch : {i} and loss : {loss}')
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save:
            # Save the model
            torch.save(self.model.state_dict(), "iris_model2.pth")

    def train_cv(self, epochs=100, lr=0.01, k_folds=5, save=True):
        # Load the data
        X, y = iris_data_xy()

        # Initialize KFold cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            print(f'\nFold {fold + 1}/{k_folds}')

            # Split the data into train and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Initialize a fresh model and optimizer for each fold
            self.model.apply(self._reset_weights)  # Reset model weights
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

            # Track the loss for each epoch
            self.epochs = epochs
            self.losses = []

            for epoch in range(self.epochs):
                # Forward pass
                y_pred = self.model.forward(X_train)
                loss = self.cross_entropy(y_pred, y_train)
                self.losses.append(loss.detach().numpy())

                # Print loss every 10 epochs
                if epoch % 10 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on the validation set
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model.forward(X_val)
                val_loss = self.cross_entropy(y_val_pred, y_val).item()
                print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
                fold_losses.append(val_loss)

        # Report the mean and standard deviation of the losses
        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        print(f'\nCross-Validation Results:')
        print(f'Mean Validation Loss: {mean_loss:.4f}')
        print(f'Standard Deviation: {std_loss:.4f}')

        if save:
            # Save the model
            torch.save(self.model.state_dict(), "iris_model-cv.pth")

    def _reset_weights(self, layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            y_eval = self.model.forward(self.X_test)
            loss = self.cross_entropy(y_eval, self.y_test).item()
            accuracy = calculate_accuracy(y_eval, self.y_test)

            print(f"Test Loss: {loss}")
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            # return model, losses

            precision, recall, f1 = calculate_metrics(y_eval, self.y_test)
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            calc_torch_roc(self.model, self.X_test, self.y_test)

    def show(self):
        # Plot the losses
        plt.plot(range(self.epochs), self.losses, label='Losses with pytorch (python)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Torch Loss Curve')
        plt.legend()
        plt.show()


class MicrogradImpl(SPNeuralNetworkImplBase):
    def __init__(self):
        super().__init__()


def load_model():
    # Load the model
    loaded_model = ModelPYTORCH()
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

    k_folds = 5

    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

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


def is_main():
    return __name__ == "__main__"


def with_torch(save=True):
    GenericModel = PyTorchImpl()
    # GenericModel.train(epochs=100 , lr=0.02)
    GenericModel.train_cv(100, 0.01, True)
    GenericModel.eval()
    GenericModel.show()


if is_main():
    with_torch()

    # with_micrograd()
