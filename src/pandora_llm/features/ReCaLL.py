from jaxtyping import Integer, Float
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class ReCaLL(FeatureComputer, LLMHandler):
    """
    Computes the ReCaLL features: the difference in log probs of text conditioned on a prefix vs. unconditional.
    Introduced by Xie et al. 2024 (https://arxiv.org/pdf/2406.15968).
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator, prefix: Integer[torch.Tensor, "tokens"]=None) -> Float[torch.Tensor, "n"]:
        """
        Computes the negative log-likelihood (NLL) feature for the given dataloader.

        Args:
            dataloader: The dataloader providing input sequences.
            accelerator: The `Accelerator` object for distributed or mixed-precision training.

        Returns:
            The NLL feature for each sequence in the dataloader.
        
        Raises:
            Exception: If the model is not loaded before calling this method.
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if prefix is not None:
            dataloader = prefix_dataloader(dataloader,prefix)
        return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="mean")
    
    @staticmethod
    def reduce(conditional_log_probs: Float[torch.Tensor, "n"], unconditional_log_probs: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n"]:
        """
        Computes conditional-unconditional log probs

        Args:
            conditional_log_probs: log probs prefixed
            unconditional_log_probs: log probs unprefixed
        Returns:
            condition-unconditional
        """
        return conditional_log_probs-unconditional_log_probs

def prefix_dataloader(dataloader: DataLoader, prefix: Integer[torch.Tensor, "tokens"]) -> DataLoader:
    """
    Prefixes the samples in the dataloader with the prefix.
    Note that prefix may push some content outside context window.

    Args:
        dataloader: token dataloader to prefix
        prefix: sequence of input ids to prefix each sample in the dataloader
    Returns:
        dataloader: dataloader prefixed with prefix
    """
    return DataLoader(torch.cat([torch.cat((prefix.repeat(sample.shape[0],1), sample), dim=1) for sample in dataloader],dim=0), batch_size=dataloader.batch_size)