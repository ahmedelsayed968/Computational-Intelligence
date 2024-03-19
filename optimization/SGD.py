from typing import List, Optional

import numpy as np
import tensorflow as tf


class SGDTrainer:
    def __init__(
        self,
        model: tf.keras.models.Model,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        epochs: int,
        loss_fn: tf.keras.losses.Loss,
        learning_schedule: Optional[List[int]] = [5, 50],
        momentum: Optional[float] = None,
    ) -> None:
        self.model = model
        self.x = x_train
        self.y = y_train
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.lr_schedule = learning_schedule
        self.momentum = momentum
        self.training_examples = self.x.shape[0]
        self.m = None

    def learning_schedule(self, t):
        return self.lr_schedule[0] / (t + self.lr_schedule[1])

    def train(self):
        losses = []

        for epoch in range(self.epochs):
            avg_loss = 0
            for point_idx in range(self.training_examples):
                random_idx = np.random.randint(0, self.training_examples)
                x_random = self.x[random_idx : random_idx + 1]
                y_random = self.y[random_idx : random_idx + 1]
                lr = self.learning_schedule(epoch * self.training_examples + point_idx)

                with tf.GradientTape(persistent=True) as t:
                    predict = self.model(x_random)
                    loss = self.loss_fn(y_random, predict)

                gradients = t.gradient(loss, self.model.trainable_variables)
                if not self.m:
                    self.m = gradients
                self._update_params(gradients, lr)

                avg_loss += loss.numpy()
                if point_idx % 10:
                    losses.append(avg_loss / 10)
                    avg_loss = 0
                    print(
                        "Epoch {}, Step {}, Loss: {:.4f}".format(
                            epoch + 1, point_idx, loss.numpy()
                        )
                    )
            return losses

    def _update_params(self, gradients, lr):
        if not self.momentum:
            for var, grad in zip(self.model.trainable_variables, gradients):
                var.assign_sub(lr * grad)
        else:
            self.m = [self.momentum * i - lr * j for i, j in zip(self.m, gradients)]
            for var, m in zip(self.model.trainable_variables, self.m):
                var.assign_add(m)
