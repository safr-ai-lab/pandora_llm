from jaxtyping import Float
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class MinK(FeatureComputer,LLMHandler):
    """
    Min-K computes token-level log probabilities for sequences and extracts the mean of the k% smallest 
    log probabilities in each sequence. Introduced by Shi et al. 2024 (https://arxiv.org/pdf/2310.16789).

    Attributes:
        model (AutoModelForCausalLM): The pre-trained language model used for log probability computation.
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n seq-1"]:
        """
        Computes token-level log probabilities for a given dataloader.

        Args:
            dataloader: A DataLoader object containing the input sequences and optional attention masks.
            accelerator: An Accelerator object to handle distributed or mixed-precision computation.

        Returns:
            A tensor of shape (n, seq-1) containing log probabilities for each token in each sequence.

        Raises:
            Exception: If the model is not loaded prior to calling this method.
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        return compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="tokens")

    @staticmethod
    def reduce(token_log_probs: Float[torch.Tensor, "n seq-1"], k: float) -> Float[torch.Tensor, "n"]:
        """
        Computes the average negative log probability of the k% rarest tokens in each sequence.

        Args:
            token_log_probs: A tensor of log probabilities for each token in the sequence, 
                            of shape (n, seq-1), where:
                            - n: number of sequences in the batch
                            - seq-1: number of tokens in each sequence (excluding the first token)
            k: A float between 0 and 1 indicating the proportion of the smallest values to select 
            in each sequence. For example, k=0.2 selects the smallest 20% of values.

        Returns:
            A tensor of shape (n,) containing the mean of the k% rarest negative log probabilities 
            for each sequence in the batch.

        Raises:
            ValueError: If k is not in the range (0, 1].
        """
        if not (0<k<=1):
            raise ValueError(f"k must be in (0,1] (got {k}).")
        sorted_log_probs, _ = torch.sort(token_log_probs, dim=1)
        num_tokens = int(k*sorted_log_probs.size(1))
        selected_values = sorted_log_probs[:,:num_tokens]
        mean_k_smallest = selected_values.sum(dim=1) / num_tokens
        return -mean_k_smallest