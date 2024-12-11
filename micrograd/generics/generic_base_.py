from sklearn.datasets import load_iris
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
from abc import ABC, abstractmethod
from micrograd.data import iris_data


import torch
import torch.nn as nn
import torch.nn.functional as F
from example_adam import epoch
from micrograd.adam import Adam
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


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




class SPNeuralNetworkImplBase(ABC):

    def __init__(self):
        ...
    @staticmethod
    def load_data():
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
        return X , y

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def eval(self):
        ...

    @abstractmethod
    def show(self):
        ...


