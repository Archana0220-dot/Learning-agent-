import numpy as np
from nn import Constant

class DeepQNetwork():
    """
    Deep Q-value Network (DQN) for reinforcement learning.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, numTrainingGames=0, learning_rate=0.001):
        self.state_size = state_dim
        self.num_actions = action_dim
        self.numTrainingGames = numTrainingGames
        self.learning_rate = learning_rate

        # Initialize weights for 1 hidden layer
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2 = np.zeros((1, action_dim))

    def _to_numpy(self, x):
        if isinstance(x, Constant):
            x = x.data
        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def run(self, states):
        x = self._to_numpy(states)
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        Q_values = np.dot(h, self.W2) + self.b2
        return Constant(Q_values)

    def get_loss(self, states, Q_target):
        Q_pred = self.run(states).data
        Q_target_np = self._to_numpy(Q_target)
        return np.mean((Q_pred - Q_target_np) ** 2)

    def gradient_update(self, states, Q_target):
        x = self._to_numpy(states)
        Q_target_np = self._to_numpy(Q_target)

        # Forward pass
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        Q_pred = np.dot(h, self.W2) + self.b2

        # Compute gradients
        grad_Q = (Q_pred - Q_target_np) / x.shape[0]
        grad_W2 = np.dot(h.T, grad_Q)
        grad_b2 = np.sum(grad_Q, axis=0, keepdims=True)

        dh = np.dot(grad_Q, self.W2.T)
        dh[h <= 0] = 0  # ReLU backward
        grad_W1 = np.dot(x.T, dh)
        grad_b1 = np.sum(dh, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
