# -*- coding: utf-8 -*-
import shutil
import subprocess
import sys
import unittest

import torch
from tests.unit.data import DATA_PATH
from torch import nn

from comet.modules.feedforward import FeedForward
from pytorch_lightning import seed_everything


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])


class TestFeedForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install("torchvision")

    @classmethod
    def tearDownClass(cls):
        uninstall("torchvision")
        # FIXME: reinstall torch==1.4.0 ??
        shutil.rmtree(DATA_PATH + "/MNIST")

    def test_MNIST(self):
        import torchvision.datasets as dsets
        import torchvision.transforms as transforms

        seed_everything(3)

        """
        STEP 1: LOADING DATASET
        """
        train_dataset = dsets.MNIST(
            root=DATA_PATH, train=True, transform=transforms.ToTensor(), download=True
        )

        test_dataset = dsets.MNIST(
            root=DATA_PATH, train=False, transform=transforms.ToTensor()
        )

        """
        STEP 2: MAKING DATASET ITERABLE
        """
        batch_size = 100
        n_iters = 600
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
            in_dim=28 * 28,
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
                images = images.view(-1, 28 * 28).requires_grad_()

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

                if iter % 500 == 0:
                    # Calculate Accuracy
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        # Load images with gradient accumulation capabilities
                        images = images.view(-1, 28 * 28).requires_grad_()

                        # Forward pass only to get logits/output
                        outputs = model(images)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted == labels).sum()

                    accuracy = 100 * correct / total
                    self.assertEqual(91, accuracy)
                    self.assertEqual(0.4050501585006714, loss.item())
