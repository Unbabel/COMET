# -*- coding: utf-8 -*-
import unittest

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

from comet.modules.feedforward import FeedForward
from pytorch_lightning import seed_everything


class TestFeedForward(unittest.TestCase):
    def test_MNIST(self):
        seed_everything(3)
        """
        STEP 1: LOADING DATASET
        """
        images, labels = load_digits(return_X_y=True)
        images = [torch.Tensor(images[i, :]) for i in range(images.shape[0])]
        labels = torch.tensor(labels, dtype=torch.long)

        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        train_dataset = list(zip(train_images, train_labels))
        test_dataset = list(zip(test_images, test_labels))

        """
        STEP 2: MAKING DATASET ITERABLE
        """
        batch_size = 256
        n_iters = 80
        num_epochs = n_iters / (len(train_dataset) / batch_size)
        num_epochs = int(num_epochs)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

        """
        STEP 3: INSTANTIATE MODEL CLASS
        """
        model = FeedForward(
            in_dim=8 * 8,
            out_dim=10,
            hidden_sizes=100,
            activations="Tanh",
            final_activation=False,
        )

        """
        STEP 4: INSTANTIATE LOSS CLASS
        """
        criterion = nn.CrossEntropyLoss()

        """
        STEP 5: INSTANTIATE OPTIMIZER CLASS
        """
        learning_rate = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        """
        STEP 7: TRAIN THE MODEL
        """
        iter = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 8 * 8).requires_grad_()

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = model(images)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                iter += 1

                if iter % 10 == 0:
                    # Calculate Accuracy
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        # Load images with gradient accumulation capabilities
                        images = images.view(-1, 8 * 8).requires_grad_()

                        # Forward pass only to get logits/output
                        outputs = model(images)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted == labels).sum()

                    accuracy = 100 * correct // total
        self.assertGreaterEqual(accuracy, 95)
        self.assertEqual(round(0.1257449835538864, 2), round(loss.item(), 2))
