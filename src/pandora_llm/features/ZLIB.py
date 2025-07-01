from jaxtyping import Integer, Num
from tqdm import tqdm
import zlib
import torch
from torch.utils.data import DataLoader
from .base import FeatureComputer

####################################################################################################
# MAIN CLASS
####################################################################################################
class ZLIB(FeatureComputer):
    """
    zlib thresholding attack
    """
    def __init__(self):
        FeatureComputer.__init__(self)

    def compute_features(self, dataloader: DataLoader) -> Num[torch.Tensor, "n"]:
        """
        Compute the negative zlib statistic for a given dataloader.

        Args:
            dataloader: input text data to compute statistic over
        Returns:
            zlib entropy of input IDs
        """
        return -compute_zlib_dl(dataloader)

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################
def compute_zlib_dl(dataloader: DataLoader) -> Integer[torch.Tensor, "n"]:
    """
    Compute the zlib cross entropy. Does not batch.

    Code taken from https://github.com/ftramer/LM_Memorization/blob/main/extraction.py

    Args:
        dataloader: input text data to compute statistic over
    Returns:
        zlib entropy of input IDs
    """
    losses = []
    for i, data_x in tqdm(enumerate(dataloader.dataset),total=len(dataloader.dataset)):
        losses.append(len(zlib.compress(bytes(data_x, 'utf-8'))))
    return torch.tensor(losses)