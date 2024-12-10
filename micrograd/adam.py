import math

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params  # List of parameters (Value objects)
        self.lr = lr          # Learning rate
        self.beta1 = beta1    # Decay rate for first moment estimates
        self.beta2 = beta2    # Decay rate for second moment estimates
        self.eps = eps        # Small value to prevent division by zero
        self.t = 0            # Time step

        # Initialize first and second moment vectors (m and v)
        self.m = {p: 0 for p in params}
        self.v = {p: 0 for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            # Update biased first moment estimate (m)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad

            # Update biased second raw moment estimate (v)
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction for first moment estimate
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)

            # Bias correction for second moment estimate
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            # Update the parameters
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0
