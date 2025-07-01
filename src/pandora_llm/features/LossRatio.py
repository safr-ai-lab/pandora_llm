from jaxtyping import Float
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class LossRatio(FeatureComputer,LLMHandler):
    """
    Computes likelihood ratio against a reference model (also known as a reference-based attack).
    Mathematically, this is log-likelihood from primary model minus log-likelihood from reference model
    
    """
    def __init__(self, model_name: str, ref_model_name: str, model_revision: str=None, model_cache_dir: str=None, ref_model_revision: str=None, ref_model_cache_dir: str=None):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, model_name, model_revision=model_revision, model_cache_dir=model_cache_dir)
        self.ref_model_name = ref_model_name
        self.ref_model_revision = ref_model_revision
        self.ref_model_cache_dir = ref_model_cache_dir

    def load_model(self, stage: str) -> None:
        """
        Loads model into memory
        
        Args:
            stage: 'primary' or 'ref'
        """
        if self.model is None:
            if stage=="primary":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            elif stage=="ref":
                self.model = AutoModelForCausalLM.from_pretrained(self.ref_model_name, revision=self.ref_model_revision, cache_dir=self.ref_model_cache_dir)
            else:
                raise Exception(f"Stage should be one of 'primary' or 'ref'. Got '{stage}'.")
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n"]:
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
        return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="mean")
    
    @staticmethod
    def reduce(primary_log_probs: Float[torch.Tensor, "n"], ref_log_probs: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n"]:
        """
        Computes loss ratio by computing primary_log_probs-ref_log_probs

        Args:
            primary_log_probs: Log probs from primary model
            ref_log_probs: Log probs from reference model
        Returns:
            primary-ref log probs
        """
        return primary_log_probs-ref_log_probs