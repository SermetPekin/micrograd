from typing import List

from micrograd.engine import Value
import math

class CrossEntropyLoss:
    @staticmethod
    def forward(logits: List[Value], target: int) -> Value:
        """
        Computes CrossEntropyLoss for a single example.

        :param logits: List of Value objects, raw outputs (logits) from the model.
        :param target: Integer index of the true class.
        :return: Loss Value.
        """
        # Step 1: Compute the exponentials of the logits
        exp_logits = [logit.exp() for logit in logits]

        # Step 2: Compute the sum of the exponentials
        sum_exp_logits = sum(exp_logits)

        # Step 3: Compute the softmax probabilities
        probs = [exp_logit / sum_exp_logits for exp_logit in exp_logits]

        # Step 4: Compute the negative log-likelihood loss for the target class
        loss = -probs[target].log()

        return loss

    @staticmethod
    def batch_forward(batch_logits: List[List[Value]], batch_targets: List[int]) -> Value:
        """
        Computes the average CrossEntropyLoss for a batch.

        :param batch_logits: List of List[Value] for all samples in the batch.
        :param batch_targets: List of true class indices for the batch.
        :return: Average loss Value.
        """
        batch_loss = sum(
            CrossEntropyLoss.forward(logits, target) for logits, target in zip(batch_logits, batch_targets)
        )
        return batch_loss / len(batch_targets)
