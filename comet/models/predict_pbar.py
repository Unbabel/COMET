import sys

import pytorch_lightning as ptl
from tqdm import tqdm


class PredictProgressBar(ptl.callbacks.progress.tqdm_progress.TQDMProgressBar):
    """Default Lightning Progress bar writes to stdout, we replace stdout with stderr"""

    def init_predict_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar
