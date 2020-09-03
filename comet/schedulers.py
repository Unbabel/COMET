# -*- coding: utf-8 -*-
r"""
Schedulers
==============
    Leraning Rate schedulers used to train COMET models.
"""
from argparse import Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantPolicy:
    """Policy for updating the LR of the ConstantLR scheduler.
    With this class LambdaLR objects became picklable.
    """

    def __call__(self, *args, **kwargs):
        return 1


class ConstantLR(LambdaLR):
    """
    Constant learning rate schedule

    Wrapper for the huggingface Constant LR Scheduler.
        https://huggingface.co/transformers/v2.1.1/main_classes/optimizer_schedules.html

    :param optimizer: torch.optim.Optimizer
    :param last_epoch:
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super(ConstantLR, self).__init__(optimizer, ConstantPolicy(), last_epoch)

    @classmethod
    def from_hparams(
        cls, optimizer: Optimizer, hparams: Namespace, **kwargs
    ) -> LambdaLR:
        """ Initializes a constant learning rate scheduler. """
        return ConstantLR(optimizer)


class WarmupPolicy:
    """Policy for updating the LR of the WarmupConstant scheduler.
    With this class LambdaLR objects became picklable.
    """

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupConstant(LambdaLR):
    """
    Warmup Linear scheduler.
    1) Linearly increases learning rate from 0 to 1 over warmup_steps
        training steps.
    2) Keeps the learning rate constant afterwards.

    :param optimizer: torch.optim.Optimizer
    :param warmup_steps: Linearly increases learning rate from 0 to 1 over warmup_steps.
    :param last_epoch:
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1
    ) -> None:
        super(WarmupConstant, self).__init__(
            optimizer, WarmupPolicy(warmup_steps), last_epoch
        )

    @classmethod
    def from_hparams(
        cls, optimizer: Optimizer, hparams: Namespace, **kwargs
    ) -> LambdaLR:
        """ Initializes a constant learning rate scheduler with warmup period. """
        return WarmupConstant(optimizer, hparams.warmup_steps)


class LinearWarmupPolicy:
    """Policy for updating the LR of the LinearWarmup scheduler.
    With this class LambdaLR objects became picklable.
    """

    def __init__(self, warmup_steps, num_training_steps):
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.warmup_steps)),
        )


class LinearWarmup(LambdaLR):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    :param optimizer: torch.optim.Optimizer
    :param warmup_steps: Linearly increases learning rate from 0 to 1*learning_rate over warmup_steps.
    :param num_training_steps: Linearly decreases learning rate from 1*learning_rate to 0. over remaining
        t_total - warmup_steps steps.
    :param last_epoch:
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        super(LinearWarmup, self).__init__(
            optimizer, LinearWarmupPolicy(warmup_steps, num_training_steps), last_epoch
        )

    @classmethod
    def from_hparams(
        cls, optimizer: Optimizer, hparams: Namespace, num_training_steps: int
    ) -> LambdaLR:
        """ Initializes a learning rate scheduler with warmup period and decreasing period. """
        return LinearWarmup(optimizer, hparams.warmup_steps, num_training_steps)


str2scheduler = {
    "linear_warmup": LinearWarmup,
    "constant": ConstantLR,
    "warmup_constant": WarmupConstant,
}
