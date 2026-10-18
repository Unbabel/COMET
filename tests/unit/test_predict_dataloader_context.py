"""Exercise predict's real DataLoader without models or accelerator hardware."""

import os
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch
from comet.models.base import CometModel
from torch.utils.data import DataLoader


@unittest.skipUnless(os.name == "posix", "MPS worker setup uses POSIX fork")
class TestPredictDataLoaderContext(unittest.TestCase):
    def test_context_matches_worker_configuration(self):
        cases = [
            # MPS availability, GPUs, requested workers, resolved workers, context
            (True, 0, None, 0, None),
            (True, 1, 0, 0, None),
            (True, 0, 2, 2, "fork"),
            (True, 1, None, 2, "fork"),
            (False, 0, None, 0, None),
            (False, 1, 0, 0, None),
            (False, 0, 2, 2, None),
            (False, 1, None, 2, None),
        ]
        samples = [{"src": "source", "mt": "translation", "ref": "reference"}]
        for mps, gpus, workers, expected_workers, expected_context in cases:
            with self.subTest(mps=mps, gpus=gpus, workers=workers):
                model = Mock()
                with ExitStack() as stack:
                    stack.enter_context(
                        patch(
                            "comet.models.base.torch.backends.mps.is_available",
                            return_value=mps,
                        )
                    )
                    loader_factory = stack.enter_context(
                        patch("comet.models.base.DataLoader", wraps=DataLoader)
                    )
                    trainer_factory = stack.enter_context(
                        patch("comet.models.base.ptl.Trainer")
                    )
                    trainer_factory.return_value.predict.return_value = [
                        {"scores": torch.tensor([0.75])}
                    ]
                    output = CometModel.predict(
                        model,
                        samples,
                        gpus=gpus,
                        num_workers=workers,
                        progress_bar=False,
                        length_batching=False,
                    )

                kwargs = loader_factory.call_args.kwargs
                self.assertEqual(kwargs["num_workers"], expected_workers)
                self.assertEqual(
                    kwargs["multiprocessing_context"], expected_context
                )
                loader = trainer_factory.return_value.predict.call_args.kwargs[
                    "dataloaders"
                ]
                self.assertIsInstance(loader, DataLoader)
                context = loader.multiprocessing_context
                self.assertEqual(
                    context.get_start_method() if context else None,
                    expected_context,
                )
                self.assertEqual(output.scores, [0.75])
                self.assertEqual(output.system_score, 0.75)


if __name__ == "__main__":
    unittest.main()
