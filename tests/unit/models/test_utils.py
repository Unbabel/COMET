import unittest

import torch

from comet.models.utils import apply_to_sample, average_pooling, mask_fill, max_pooling


class TestUtils(unittest.TestCase):
    def test_mask_fill(self):
        tokens = torch.tensor([[2, 3, 2], [1, 2, 0]])
        embeddings = torch.tensor(
            [
                [
                    [3.1416, 3.1416, 3.1416],
                    [4.1416, 4.1416, 4.1416],
                    [3.1416, 3.1416, 3.1416],
                ],
                [
                    [1.1416, 1.1416, 1.1416],
                    [3.1416, 3.1416, 3.1416],
                    [0.0000, 0.0000, 0.0000],
                ],
            ]
        )

        expected = torch.tensor(
            [
                [
                    [3.1416, 3.1416, 3.1416],
                    [4.1416, 4.1416, 4.1416],
                    [3.1416, 3.1416, 3.1416],
                ],
                [
                    [1.1416, 1.1416, 1.1416],
                    [3.1416, 3.1416, 3.1416],
                    [-1.000, -1.000, -1.000],
                ],
            ]
        )
        result = mask_fill(-1, tokens, embeddings, 0)
        self.assertTrue(torch.equal(result, expected))

        expected = torch.tensor(
            [
                [
                    [3.1416, 3.1416, 3.1416],
                    [10.000, 10.000, 10.000],
                    [3.1416, 3.1416, 3.1416],
                ],
                [
                    [1.1416, 1.1416, 1.1416],
                    [3.1416, 3.1416, 3.1416],
                    [-1.000, -1.000, -1.000],
                ],
            ]
        )
        result = mask_fill(10.000, tokens, embeddings, 3)
        self.assertTrue(torch.equal(result, expected))

    def test_average_pooling(self):
        tokens = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 0]])
        embeddings = torch.tensor(
            [
                [
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                ],
                [
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                    [0.0000, 0.0000, 0.0000],
                ],
            ]
        )
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

        expected = torch.tensor([[3.1416, 3.1416, 3.1416], [3.1416, 3.1416, 3.1416]])
        result = average_pooling(tokens, embeddings, mask, 0)
        self.assertTrue(torch.equal(result, expected))

    def test_max_pooling(self):
        tokens = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 0]])
        embeddings = torch.tensor(
            [
                [
                    [3.1416, 4.1416, 5.1416],
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                    [3.1416, 3.1416, 3.1416],
                ],
                [
                    [6.1416, 3.1416, -3.1416],
                    [3.1416, 3.1416, -1.1416],
                    [3.1416, 7.1416, -3.1416],
                    [0.0000, 0.0000, 0.0000],
                ],
            ]
        )
        expected = torch.tensor([[3.1416, 4.1416, 5.1416], [6.1416, 7.1416, -1.1416]])
        result = max_pooling(tokens, embeddings, 0)
        self.assertTrue(torch.equal(result, expected))

    def test_apply_to_sample(self):
        """ If this works then move_to_cuda and move_to_cpu should work. """

        def to_float(tensor):
            return tensor.float()

        def to_int(tensor):
            return tensor.int()

        sample = {
            "tensor": torch.tensor([2, 2, 2, 2]),
            "another_tensor": torch.tensor([1, 2, 1, 2]),
        }

        sample = apply_to_sample(to_float, sample)
        self.assertTrue(sample["tensor"].is_floating_point())
        self.assertTrue(sample["another_tensor"].is_floating_point())

        sample = apply_to_sample(to_int, sample)
        self.assertFalse(sample["tensor"].is_floating_point())
        self.assertFalse(sample["another_tensor"].is_floating_point())

        sample = [torch.tensor([2, 2, 2, 2]), torch.tensor([1, 2, 1, 2])]
        sample = apply_to_sample(to_float, sample)
        self.assertTrue(sample[0].is_floating_point())
        self.assertTrue(sample[1].is_floating_point())

        sample = apply_to_sample(to_int, sample)
        self.assertFalse(sample[0].is_floating_point())
        self.assertFalse(sample[1].is_floating_point())

        sample = torch.tensor([2, 2, 2, 2])
        sample = apply_to_sample(to_float, sample)
        self.assertTrue(sample.is_floating_point())
        sample = apply_to_sample(to_int, sample)
        self.assertFalse(sample.is_floating_point())
