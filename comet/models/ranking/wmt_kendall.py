from torchmetrics import Metric
import torch


class WMTKendall(Metric):
    def __init__(self, dist_sync_on_step=False, prefix=""):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("concordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("discordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.prefix = prefix
        
    def update(self, distance_pos: torch.Tensor, distance_neg: torch.Tensor):
        assert distance_pos.shape == distance_neg.shape
        self.concordance = torch.sum((distance_pos < distance_neg).float())
        self.discordance = torch.sum((distance_pos >= distance_neg).float())

    def compute(self):
        return {
            self.prefix + "_kendall":  (self.concordance - self.discordance) / (self.concordance + self.discordance)
        }