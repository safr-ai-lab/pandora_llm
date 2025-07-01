from jaxtyping import Float
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl
from .MinK import MinK

####################################################################################################
# MAIN CLASS
####################################################################################################
class MinKXX(FeatureComputer,LLMHandler):
    """
    Min-K++ analyzes token-level  log probabilities in sequences by comparing the predicted token's log probability 
    to the distribution of all possible next tokens. Introduced by Zhang et al. 2024 (https://arxiv.org/pdf/2404.02936).

    Attributes:
        model (AutoModelForCausalLM): The pre-trained language model used for log probability computation.
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n seq-1 3"]:
        """
        Computes predicted log probabilities for all tokens in the vocabulary at each sequence position,
        concatenating the target token's log probability as the 0th index of the vocabulary dimension.

        Args:
            dataloader: A DataLoader object containing the input sequences and optional attention masks.
            accelerator: An  Accelerator object to handle distributed or mixed-precision computation.

        Returns:
            A tensor of shape (n, seq-1, vocab+1), where n is the number of sequences in the batch, 
            seq-1 is the number of tokens in each sequence (excluding the first token), 
            and vocab+1 is the vocabulary size plus one allocated for the target.
        
        Raises:
            Exception: If the model is not loaded before calling this method.
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        return compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="tokens+z")

    @staticmethod
    def reduce(log_probs_w_z: Float[torch.Tensor, "n seq-1 3"], k: float) -> Float[torch.Tensor, "n"]:
        """
        Computes the average negative log probability of the k% rarest log probs
        (z-scored against other possible next tokens) in each sequence.

        Args:
            log_probs: A tensor of shape (n, seq-1, 3) containing:
                The target token's log probability as the 0th index of the last dimension.
                The mean and stdev log probs as the 1st and 2nd index of the last dimension.
            k: A float between 0 and 1, indicating the proportion of the rarest values 
               to consider in each sequence. For example, k=0.2 selects the rarest 20% of values.

        Returns:
            A tensor of shape (n,) containing the mean of the k% rarest z-scored log probabilities 
            for each sequence in the batch.

        Raises:
            ValueError: If k is not in the range (0, 1].
        """
        log_probs_z = (log_probs_w_z[:,:,0]-log_probs_w_z[:,:,1])/(log_probs_w_z[:,:,2]+1e-8)
        return MinK.reduce(log_probs_z, k=k)