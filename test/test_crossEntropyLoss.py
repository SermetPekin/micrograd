import pytest
import torch
import torch.nn as nn
from micrograd.engine import Value
from micrograd.cross_entropy import CrossEntropyLoss  # Assuming this is your custom implementation

def test_micrograd_vs_torch_cross_entropy():
    # Define the logits and targets
    logits_micrograd = [[Value(-1.0), Value(-2.0), Value(-3.0)],
                        [Value(0.5), Value(-1.5), Value(-0.5)]]
    targets_micrograd = [0, 2]

    # Torch equivalent tensors
    logits_torch = torch.tensor([[-1.0, -2.0, -3.0],
                                 [0.5, -1.5, -0.5]], dtype=torch.float32)
    targets_torch = torch.tensor([0, 2], dtype=torch.long)

    # Compute micrograd loss
    loss_micrograd = CrossEntropyLoss.batch_forward(logits_micrograd, targets_micrograd)
    micrograd_loss_value = loss_micrograd.data

    # Compute torch loss
    criterion = nn.CrossEntropyLoss()
    loss_torch = criterion(logits_torch, targets_torch)
    torch_loss_value = loss_torch.item()

    # Print losses for debugging
    print(f"Micrograd Loss: {micrograd_loss_value:.4f}")
    print(f"Torch Loss: {torch_loss_value:.4f}")

    # Assert that the losses are approximately equal
    assert abs(micrograd_loss_value - torch_loss_value) < 1e-4, \
        f"Losses do not match: Micrograd={micrograd_loss_value}, Torch={torch_loss_value}"
