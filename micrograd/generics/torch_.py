from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from .generic_base_ import SPNeuralNetworkImplBase ,get_iris_data_split, iris_data_xy


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


# PyTorch Implementation Class
class PyTorchImpl(SPNeuralNetworkImplBase):
    def __init__(self):
        super().__init__()
        self.model = ModelPYTORCH()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.epochs = 100
        self.losses = []

    def __call__(self, *args, **kwargs):
        self.train()
        self.eval()
        self.show()

    # Standard training method
    def train(self, epochs=100, lr=0.01, save=True):
        self.epochs = epochs
        self.losses = []

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.forward(self.X_train)
            loss = self.cross_entropy(y_pred, self.y_train)
            self.losses.append(loss.detach().numpy())

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save:
            torch.save(self.model.state_dict(), "iris_model.pth")



    # Cross-validation training method
    def train_cv(self, epochs=100, lr=0.01, k_folds=5, save=True):
        # X, y = iris_data_xy()
        X , y = self.load_data()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f'\nFold {fold + 1}/{k_folds}')

            # Split data into training and validation sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Reset model weights
            self.model.apply(self._reset_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.losses = []

            for epoch in range(epochs):
                # Forward pass
                y_pred = self.model.forward(X_train)
                loss = self.cross_entropy(y_pred, y_train)
                self.losses.append(loss.detach().numpy())

                if epoch % 10 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

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

        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        print(f'\nCross-Validation Results:')
        print(f'Mean Validation Loss: {mean_loss:.4f}')
        print(f'Standard Deviation: {std_loss:.4f}')

        if save:
            torch.save(self.model.state_dict(), "iris_model_cv.pth")

    # Reset model weights
    def _reset_weights(self, layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()



    # Visualization method
    def show(self):
        plt.plot(range(self.epochs), self.losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()


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

def load_model_torch():
    # Load the model
    loaded_model = ModelPYTORCH()
    loaded_model.load_state_dict(torch.load("iris_model.pth"))
    loaded_model.eval()
    return loaded_model




def with_torch(epochs=100, lr=0.01, k_folds=5):
    GenericModel = PyTorchImpl()
    # GenericModel.train(epochs=100 , lr=0.02)
    GenericModel.train_cv(epochs, lr, k_folds)
    # GenericModel.eval()
    GenericModel.show()
