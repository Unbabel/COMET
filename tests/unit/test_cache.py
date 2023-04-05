# -*- coding: utf-8 -*-
import unittest

import torch

from comet.models.lru_cache import tensor_lru_cache


class TestLRUCache(unittest.TestCase):
    @tensor_lru_cache(None)  # unlimited size
    def add(self, x, y):
        return x + y

    def test_cache(self):
        for _ in range(10):
            tmp = self.add(
                torch.tensor([[0, 1, 2], [2, 3, 4]]),
                torch.tensor([[5, 6, 7], [8, 9, 10]]),
            )
        cache_info = self.add.cache_info()
        assert cache_info.hits == 9
        self.add.cache_clear()

        for _ in range(10):
            tmp = self.add(torch.tensor([[0, 2]]), torch.tensor([[5, 8]]))
        cache_info = self.add.cache_info()
        assert cache_info.hits == 9
        self.add.cache_clear()

        for _ in range(10):
            tmp = self.add(torch.tensor([0, 2]), torch.tensor([5, 8]))
        cache_info = self.add.cache_info()
        assert cache_info.hits == 9
        self.add.cache_clear()

        with self.assertRaises(Exception) as context:
            for _ in range(10):
                tmp = self.add(torch.tensor(0), torch.tensor(1))

        self.assertTrue(
            "Tensor needs to be at least 1-Dimensional." in str(context.exception)
        )
