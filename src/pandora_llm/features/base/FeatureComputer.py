from abc import ABC, abstractmethod
from jaxtyping import Num
import torch
from torch.utils.data import DataLoader

class FeatureComputer(ABC):
    """
    Base class for a feature computer.
    This class requires a compute_features function, and provides default plotting functions.
    """
    @abstractmethod
    def compute_features(self, dataloader: DataLoader, **kwargs) -> Num[torch.Tensor, "n ..."]:
        """
        This method should be implemented by subclasses to compute the features for the given dataloader.

        Args:
            dataloader: input data to compute features over
        Returns:
            Tensor of features computed on the input dataloader
        """
        raise NotImplementedError()