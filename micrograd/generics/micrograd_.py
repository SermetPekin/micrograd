import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from micrograd import Value, MLP
from micrograd.adam import Adam
from sklearn.utils import shuffle as shuffle_data_sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

from micrograd.generics.generic_base_ import SPNeuralNetworkImplBase


def plot_training_and_validation_loss(training_losses, validation_losses, title="Training and Validation Loss"):
    min_length = min(len(training_losses), len(validation_losses))
    training_losses = training_losses[:min_length]
    validation_losses = validation_losses[:min_length]
    epochs = range(1, min_length + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to plot training and validation losses
def plot_training_and_validation_loss(training_losses, validation_losses, title="Training and Validation Loss"):
    min_length = min(len(training_losses), len(validation_losses))
    training_losses = training_losses[:min_length]
    validation_losses = validation_losses[:min_length]
    epochs = range(1, min_length + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Updated MicrogradImpl class
class MicrogradImpl(SPNeuralNetworkImplBase):
    def __init__(self, in_feats=4, hidden_layers=[7, 7], out_feats=3, lr=0.01, epochs=100, k_folds=2):
        self.in_feats = in_feats
        self.hidden_layers = hidden_layers
        self.out_feats = out_feats
        self.lr = lr
        self.epochs = epochs
        self.k_folds = k_folds

        # Initialize the model and optimizer
        self.model = MLP(in_feats, hidden_layers + [out_feats])
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.losses = []
        self.validation_losses = []

    def load_data(self):
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def train(self, X_train, y_train, verbose=True):
        """Train the model with the given training data."""
        self.losses = []
        X_train, y_train = shuffle_data_sklearn(X_train, y_train)

        # One-hot encode y_train
        y_train_onehot = np.zeros((y_train.shape[0], self.out_feats))
        y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for i in range(len(X_train)):
                # Forward pass
                inputs = [Value(x) for x in X_train[i]]
                targets = [Value(y) for y in y_train_onehot[i]]
                outputs = self.model(inputs)

                # Calculate Cross-Entropy Loss
                exp_outputs = [o.exp() for o in outputs]
                sum_exp_outputs = sum(exp_outputs)
                probs = [o / sum_exp_outputs for o in exp_outputs]
                loss = -sum(t * p.log() for t, p in zip(targets, probs))

                epoch_loss += loss.data

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / len(X_train)
            self.losses.append(avg_epoch_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

    def train_cv(self, X, y, k_folds=None, shuffle=True, verbose=True):
        """Perform k-fold cross-validation."""
        X, y = shuffle_data_sklearn(X, y)
        if k_folds is None:
            k_folds = self.k_folds

        kf = KFold(n_splits=k_folds, shuffle=shuffle, random_state=42)
        self.validation_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            if verbose:
                print(f"\nFold {fold + 1}/{k_folds}")

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Reset model and optimizer before each fold
            self.model = MLP(self.in_feats, self.hidden_layers + [self.out_feats])
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

            # Train the model on the current fold
            self.train(X_train, y_train, verbose=False)

            # Evaluate on the validation set
            val_loss = self.eval(X_val, y_val, print_results=False)
            if verbose:
                print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")
            self.validation_losses.append(val_loss)

        mean_loss = np.mean(self.validation_losses)
        std_loss = np.std(self.validation_losses)
        print(f"\nCross-Validation Results:")
        print(f"Mean Validation Loss: {mean_loss:.4f}")
        print(f"Standard Deviation: {std_loss:.4f}")

    def eval(self, X_test, y_test, print_results=True):
        """Evaluate the model on the test data."""
        correct = 0
        total = len(X_test)
        total_loss = 0.0
        self.eval_fold_losses = []

        # One-hot encode y_test
        y_test_onehot = np.zeros((y_test.shape[0], self.out_feats))
        y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

        for i in range(total):
            inputs = [Value(x) for x in X_test[i]]
            targets = [Value(y) for y in y_test_onehot[i]]
            outputs = self.model(inputs)

            # Calculate Cross-Entropy Loss
            exp_outputs = [o.exp() for o in outputs]
            sum_exp_outputs = sum(exp_outputs)
            probs = [o / sum_exp_outputs for o in exp_outputs]
            loss = -sum(t * p.log() for t, p in zip(targets, probs))

            total_loss += loss.data
            self.eval_fold_losses.append(loss.data)

            predicted = np.argmax([o.data for o in outputs])
            if predicted == y_test[i]:
                correct += 1

        avg_loss = total_loss / total
        accuracy = correct / total

        if print_results:
            print(f"\nTest Loss: {avg_loss:.4f}")
            print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return avg_loss

    def show(self):
        """Plot the loss curve."""
        plot_training_and_validation_loss(self.losses, self.eval_fold_losses)


def calculate_accuracy(_y_pred, _y_true):
    y_pred_classes = torch.argmax(_y_pred, axis=1)  # Get the predicted class
    acc = (y_pred_classes == _y_true).sum().item() / len(_y_true)
    return acc


def calculate_metrics(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    precision = precision_score(y_true, predicted_classes, average='weighted')
    recall = recall_score(y_true, predicted_classes, average='weighted')
    f1 = f1_score(y_true, predicted_classes, average='weighted')
    return precision, recall, f1


def with_micrograd(epochs=100, lr=0.01, k_folds=2):
    # GenericModel = MicrogradImpl()
    # GenericModel.train_cv(epochs, lr, k_folds)
    # # GenericModel.eval()
    # GenericModel.show()
    from sklearn.model_selection import train_test_split
    model = MicrogradImpl(epochs=epochs, lr=lr, k_folds=k_folds)

    X, y = model.load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    model.train_cv(X_train, y_train, k_folds=k_folds)
    model.eval(X_test, y_test)
    model.show()
