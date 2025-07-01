import math
from jaxtyping import Float
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl
from .ZLIB import compute_zlib_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class LossZLIB(FeatureComputer,LLMHandler):
    """
    Loss Ratio with ZLIB entropy as the reference
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self,*args,**kwargs)

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator=None, mode: str="primary") -> Float[torch.Tensor, "n"]:
        """
        Compute the specified component of the loss ratio

        Args:
            dataloader: input data to compute features over
            accelerator: accelerator object
            mode: either "primary" or "ref"
        Returns:
            Either log probs from model if mode="primary" or zlib entropy if mode="ref"
        Raises:
            ValueError: if mode is not one of "primary" or "ref"
        """
        if mode=="primary":
            if self.model is None:
                raise Exception("Please call .load_model() to load the model first.")
            return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator)
        elif mode=="ref":
            return -compute_zlib_dl(dataloader)
        else:
            raise ValueError(f"Expect 'mode' to be one of 'primary' or 'ref' (got {mode})")
    
    @staticmethod
    def reduce(primary_log_probs: Float[torch.Tensor, "n"], zlib_entropy: Float[torch.Tensor, "n"])  -> Float[torch.Tensor, "n"]:
        """
        Performs the ratio

        Args:
            primary_log_probs: log probs from model
            zlib_entropy: zlib entropy
        Returns:
            primary_log_probs-zlib_entropy
        """
        return primary_log_probs-zlib_entropy