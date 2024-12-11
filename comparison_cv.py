# from sklearn.datasets import load_iris
# import pandas as pd
# from typing import Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#
# from example_adam import epoch
# from micrograd.data import iris_data
# from micrograd.adam import Adam
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import KFold
#
#
# def process_data_for_torch_and_micrograd(df: pd.DataFrame):
#     X = df.drop('variety', axis=1)
#     y = df['variety']
#     X = X.values
#     y = y.values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     X_train = torch.FloatTensor(X_train)
#     X_test = torch.FloatTensor(X_test)
#     y_train = torch.LongTensor(y_train)
#     y_test = torch.LongTensor(y_test)
#     return X_train, X_test, y_train, y_test
#
#
# def get_iris_data_split() -> Tuple['X_train', 'X_test', 'y_train', 'y_test']:
#     df = iris_data()
#     return process_data_for_torch_and_micrograd(df)
#
#
# from abc import ABC, abstractmethod
#
#
# class SPNeuralNetworkImplBase(ABC):
#     data_fnc = get_iris_data_split
#
#     def __init__(self):
#         ...
#
#     @abstractmethod
#     def set_data_fnc(self):
#         self.X_train, self.X_test, self.y_train, self.y_test = self.data_fnc()
#
#     @abstractmethod
#     def train(self):
#         ...
#
#     @abstractmethod
#     def eval(self):
#         ...
#
#     @abstractmethod
#     def show(self):
#         ...
#
#
# class ModelPYTORCH(nn.Module):
#     def __init__(self, in_feats: int = 4, out_feats: int = 3, hidden1=7, hidden2=7):
#         super(ModelPYTORCH, self).__init__()
#         self.fc1 = nn.Linear(in_feats, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.out = nn.Linear(hidden2, out_feats)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.out(x)
#         return x
#
#
# def iris_data_xy():
#     from sklearn.datasets import load_iris
#     from sklearn.preprocessing import StandardScaler
#     data = load_iris()
#     X = data.data
#     y = data.target
#
#     # Standardize the dataset
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#
#     # Convert to PyTorch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.long)
#
#     return X_tensor, y_tensor
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
#
#
# # Function to load and preprocess the Iris dataset
# def iris_data_xy():
#     data = load_iris()
#     X, y = data.data, data.target
#
#     # Standardize the dataset
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#
#     # Convert to PyTorch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.long)
#
#     return X_tensor, y_tensor
#
#
# # Base class (assuming it's defined elsewhere)
# class SPNeuralNetworkImplBase:
#     pass
#
#
# # PyTorch Implementation Class
# class PyTorchImpl(SPNeuralNetworkImplBase):
#     def __init__(self):
#         super().__init__()
#         self.model = ModelPYTORCH()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
#         self.cross_entropy = nn.CrossEntropyLoss()
#         self.epochs = 100
#         self.losses = []
#
#     def __call__(self, *args, **kwargs):
#         self.train()
#         self.eval()
#         self.show()
#
#     # Standard training method
#     def train(self, epochs=100, lr=0.01, save=True):
#         self.epochs = epochs
#         self.losses = []
#
#         optimizer = optim.Adam(self.model.parameters(), lr=lr)
#
#         for epoch in range(epochs):
#             # Forward pass
#             y_pred = self.model.forward(self.X_train)
#             loss = self.cross_entropy(y_pred, self.y_train)
#             self.losses.append(loss.detach().numpy())
#
#             # Print loss every 10 epochs
#             if epoch % 10 == 0:
#                 print(f'Epoch: {epoch}, Loss: {loss:.4f}')
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if save:
#             torch.save(self.model.state_dict(), "iris_model.pth")
#
#     # Cross-validation training method
#     def train_cv(self, epochs=100, lr=0.01, k_folds=5, save=True):
#         X, y = iris_data_xy()
#         kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#         fold_losses = []
#
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#             print(f'\nFold {fold + 1}/{k_folds}')
#
#             # Split data into training and validation sets
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
#
#             # Reset model weights
#             self.model.apply(self._reset_weights)
#             optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             self.losses = []
#
#             for epoch in range(epochs):
#                 # Forward pass
#                 y_pred = self.model.forward(X_train)
#                 loss = self.cross_entropy(y_pred, y_train)
#                 self.losses.append(loss.detach().numpy())
#
#                 if epoch % 10 == 0:
#                     print(f'Epoch: {epoch}, Loss: {loss:.4f}')
#
#                 # Backpropagation
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             # Evaluate on the validation set
#             self.model.eval()
#             with torch.no_grad():
#                 y_val_pred = self.model.forward(X_val)
#                 val_loss = self.cross_entropy(y_val_pred, y_val).item()
#                 print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
#                 fold_losses.append(val_loss)
#
#         mean_loss = np.mean(fold_losses)
#         std_loss = np.std(fold_losses)
#         print(f'\nCross-Validation Results:')
#         print(f'Mean Validation Loss: {mean_loss:.4f}')
#         print(f'Standard Deviation: {std_loss:.4f}')
#
#         if save:
#             torch.save(self.model.state_dict(), "iris_model_cv.pth")
#
#     # Reset model weights
#     def _reset_weights(self, layer):
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()
#
#     # # Evaluation method
#     # def eval(self):
#     #     self.model.eval()
#     #     with torch.no_grad():
#     #         y_eval = self.model.forward(self.X_test)
#     #         loss = self.cross_entropy(y_eval, self.y_test).item()
#     #         accuracy = calculate_accuracy(y_eval, self.y_test)
#     #
#     #         print(f"Test Loss: {loss:.4f}")
#     #         print(f"Test Accuracy: {accuracy * 100:.2f}%")
#     #
#     #         precision, recall, f1 = calculate_metrics(y_eval, self.y_test)
#     #         print(f"Precision: {precision:.2f}")
#     #         print(f"Recall: {recall:.2f}")
#     #         print(f"F1-Score: {f1:.2f}")
#     #
#     #         calc_torch_roc(self.model, self.X_test, self.y_test)
#
#     # Visualization method
#     def show(self):
#         plt.plot(range(self.epochs), self.losses, label='Training Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.title('Training Loss Curve')
#         plt.legend()
#         plt.show()
#
#
# # class MicrogradImpl(SPNeuralNetworkImplBase):
# #     def __init__(self):
# #         super().__init__()
#
#
# from micrograd import Value
# from micrograd import MLP
#
# import numpy as np
# import matplotlib.pyplot as plt
# from micrograd import Value, MLP
# from sklearn.model_selection import KFold
# from micrograd.adam import Adam
#
# from sklearn.utils import shuffle
#
#
# def shuffle_data_sklearn(X, y):
#     """Shuffle X and y using scikit-learn's shuffle function."""
#     X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
#     return X_shuffled, y_shuffled
#
# def plot_training_and_validation_loss(training_losses, validation_losses, title="Training and Validation Loss"):
#     min_length = min(len(training_losses), len(validation_losses))
#     training_losses = training_losses[:min_length]
#     validation_losses = validation_losses[:min_length]
#     epochs = range(1, min_length + 1)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, training_losses, label='Training Loss', marker='o')
#     plt.plot(epochs, validation_losses, label='Validation Loss', marker='x')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# from micrograd import Value, MLP
# from micrograd.adam import Adam
# from sklearn.utils import shuffle as shuffle_data_sklearn
#
# # Function to plot training and validation losses
# def plot_training_and_validation_loss(training_losses, validation_losses, title="Training and Validation Loss"):
#     min_length = min(len(training_losses), len(validation_losses))
#     training_losses = training_losses[:min_length]
#     validation_losses = validation_losses[:min_length]
#     epochs = range(1, min_length + 1)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, training_losses, label='Training Loss', marker='o')
#     plt.plot(epochs, validation_losses, label='Validation Loss', marker='x')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# # Updated MicrogradImpl class
# class MicrogradImpl:
#     def __init__(self, in_feats=4, hidden_layers=[7, 7], out_feats=3, lr=0.01, epochs=100):
#         self.in_feats = in_feats
#         self.hidden_layers = hidden_layers
#         self.out_feats = out_feats
#         self.lr = lr
#         self.epochs = epochs
#
#         # Initialize the model and optimizer
#         self.model = MLP(in_feats, hidden_layers + [out_feats])
#         self.optimizer = Adam(self.model.parameters(), lr=lr)
#         self.losses = []
#         self.validation_losses = []
#
#     def train(self, X_train, y_train, verbose=True):
#         """Train the model with the given training data."""
#         self.losses = []
#         X_train, y_train = shuffle_data_sklearn(X_train, y_train)
#
#         # One-hot encode y_train
#         y_train_onehot = np.zeros((y_train.shape[0], self.out_feats))
#         y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
#
#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#
#             for i in range(len(X_train)):
#                 # Forward pass
#                 inputs = [Value(x) for x in X_train[i]]
#                 targets = [Value(y) for y in y_train_onehot[i]]
#                 outputs = self.model(inputs)
#
#                 # Calculate Cross-Entropy Loss
#                 exp_outputs = [o.exp() for o in outputs]
#                 sum_exp_outputs = sum(exp_outputs)
#                 probs = [o / sum_exp_outputs for o in exp_outputs]
#                 loss = -sum(t * p.log() for t, p in zip(targets, probs))
#
#                 epoch_loss += loss.data
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#             avg_epoch_loss = epoch_loss / len(X_train)
#             self.losses.append(avg_epoch_loss)
#
#             if verbose and epoch % 10 == 0:
#                 print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
#
#     def train_cv(self, X, y, k_folds=5, verbose=True):
#         """Perform k-fold cross-validation."""
#         X, y = shuffle_data_sklearn(X, y)
#
#         kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#         self.validation_losses = []
#
#         for fold, (train_index, val_index) in enumerate(kf.split(X)):
#             print(f"\nFold {fold + 1}/{k_folds}")
#
#             X_train, X_val = X[train_index], X[val_index]
#             y_train, y_val = y[train_index], y[val_index]
#
#             # Reset model and optimizer before each fold
#             self.model = MLP(self.in_feats, self.hidden_layers + [self.out_feats])
#             self.optimizer = Adam(self.model.parameters(), lr=self.lr)
#
#             # Train the model on the current fold
#             self.train(X_train, y_train, verbose=False)
#
#             # Evaluate on the validation set
#             val_loss = self.eval(X_val, y_val, print_results=False)
#             print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")
#             self.validation_losses.append(val_loss)
#
#         mean_loss = np.mean(self.validation_losses)
#         std_loss = np.std(self.validation_losses)
#         print(f"\nCross-Validation Results:")
#         print(f"Mean Validation Loss: {mean_loss:.4f}")
#         print(f"Standard Deviation: {std_loss:.4f}")
#
#     def eval(self, X_test, y_test, print_results=True):
#         """Evaluate the model on the test data."""
#         correct = 0
#         total = len(X_test)
#         total_loss = 0.0
#         self.eval_fold_losses = []
#
#         # One-hot encode y_test
#         y_test_onehot = np.zeros((y_test.shape[0], self.out_feats))
#         y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
#
#         for i in range(total):
#             inputs = [Value(x) for x in X_test[i]]
#             targets = [Value(y) for y in y_test_onehot[i]]
#             outputs = self.model(inputs)
#
#             # Calculate Cross-Entropy Loss
#             exp_outputs = [o.exp() for o in outputs]
#             sum_exp_outputs = sum(exp_outputs)
#             probs = [o / sum_exp_outputs for o in exp_outputs]
#             loss = -sum(t * p.log() for t, p in zip(targets, probs))
#
#             total_loss += loss.data
#             self.eval_fold_losses.append(loss.data)
#
#             predicted = np.argmax([o.data for o in outputs])
#             if predicted == y_test[i]:
#                 correct += 1
#
#         avg_loss = total_loss / total
#         accuracy = correct / total
#
#         if print_results:
#             print(f"\nTest Loss: {avg_loss:.4f}")
#             print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
#         return avg_loss
#
#     def show(self):
#         """Plot the loss curve."""
#         plot_training_and_validation_loss(self.losses, self.eval_fold_losses)
#
#
#
# def load_model():
#     # Load the model
#     loaded_model = ModelPYTORCH()
#     loaded_model.load_state_dict(torch.load("iris_model.pth"))
#     loaded_model.eval()
#     return loaded_model
#
#
# # def calculate_accuracy(_y_pred, _y_true):
# #     y_pred_classes = torch.argmax(_y_pred, axis=1)  # Get the predicted class
# #     acc = (y_pred_classes == _y_true).sum().item() / len(_y_true)
# #     return acc
#
#
# from sklearn.metrics import precision_score, recall_score, f1_score
#
#
# # def calculate_metrics(y_pred, y_true):
# #     predicted_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
# #     y_true = y_true.cpu().numpy()
# #
# #     precision = precision_score(y_true, predicted_classes, average='weighted')
# #     recall = recall_score(y_true, predicted_classes, average='weighted')
# #     f1 = f1_score(y_true, predicted_classes, average='weighted')
# #
# #     return precision, recall, f1
#
#
# def calc_torch_roc(model, X_test, y_test):
#     # Evaluate the model
#     model.eval()
#     with torch.no_grad():
#         y_logits = model(X_test)  # Get raw logits
#         y_probs = torch.softmax(y_logits, dim=1)  # Convert logits to probabilities
#
#     # Convert predictions and targets to numpy arrays
#     y_probs_np = y_probs.numpy()
#     y_test_np = y_test.numpy()
#
#     # Calculate ROC AUC score
#     roc_auc = roc_auc_score(
#         y_test_np,
#         y_probs_np,
#         multi_class='ovr',  # 'ovr' for one-vs-rest (multi-class)
#         average='weighted'  # Weighted average
#     )
#
#     print(f"ROC AUC Score: {roc_auc:.4f}")
#
#
# def is_main():
#     return __name__ == "__main__"
#
#
# def with_micrograd(epochs=100, lr=0.01, k_folds=5):
#     # GenericModel = MicrogradImpl()
#     # GenericModel.train_cv(epochs, lr, k_folds)
#     # # GenericModel.eval()
#     # GenericModel.show()
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split
#
#     # Load Iris dataset
#     data = load_iris()
#     X = data.data
#     y = data.target
#
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Convert to NumPy arrays for compatibility
#     X_train, X_test = np.array(X_train), np.array(X_test)
#     y_train, y_test = np.array(y_train), np.array(y_test)
#
#     # Initialize and train the model
#     model = MicrogradImpl()
#     model.train_cv(X_train, y_train, k_folds=5)
#     model.eval(X_test, y_test)
#     model.show()
#
#
# def with_torch(epochs=100, lr=0.01, k_folds=5):
#     GenericModel = PyTorchImpl()
#     # GenericModel.train(epochs=100 , lr=0.02)
#     GenericModel.train_cv(epochs, lr, k_folds)
#     # GenericModel.eval()
#     GenericModel.show()
#
#
# if is_main():
#     # with_torch()
#
#     with_micrograd()
