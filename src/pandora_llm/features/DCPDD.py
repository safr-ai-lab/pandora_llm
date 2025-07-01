from jaxtyping import Float, Integer
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class DCPDD(FeatureComputer, LLMHandler):
    """
    Computes the divergence between vocab probability distribution and random internet text.
    Introduced by Zhang et al. 2024 (https://arxiv.org/pdf/2409.14781).
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)
    
    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator=None, tokenizer: AutoTokenizer=None, mode: str="primary", smoothing: str="laplace") -> Float[torch.Tensor, "n seq-1"]:
        """
        Compute either token-level log probs (mode="primary") or reference frequency-based probability (mode="ref")

        Args:
            dataloader: input data to compute statistic over
            accelerator: accelerator object
        Returns:
            Vocab-level log probs
        Raises:
            Exception: if model was not loaded
            ValueError
        """
        if mode=="primary":
            if self.model is None:
                raise Exception("Please call .load_model() to load the model first.")
            return compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="tokens")
        elif mode=="ref":
            return compute_ref_probs(dataloader=dataloader, tokenizer=tokenizer, smoothing=smoothing)
        else:
            raise ValueError(f"Expected mode to be 'primary' or 'ref' (got {mode}).")

    @staticmethod
    def reduce(dataloader: DataLoader, target_log_probs: Float[torch.Tensor, "n seq-1"], ref_probs: Integer[torch.Tensor, "vocab"], score_bound: float=0.01) -> Float[torch.Tensor, "n"]:
        """
        Computes divergence between target log probs and reference frequency, applying a score_bound and averaging over the input_ids that are the first occurence in the sequence.

        Args:
            dataloader: dataloader that produced the target_log_probs
            target_log_probs: tensor of next token probabilities
            ref_probs: reference probability of vocab
            score_bound: upper bound on divergence score
        Returns:
            Tensor of divergence (cross-entropy) scores
        """
        input_ids = torch.cat([batch["input_ids"] if isinstance(batch, dict) else batch for batch in dataloader],dim=0)[:,1:]
        divergence = -torch.exp(target_log_probs)*torch.log(ref_probs[input_ids])
        divergence = torch.clamp(divergence, max=score_bound)
        # print("Div",divergence.shape)
        # for seq in input_ids:
        #     print(type(seq),len(seq))
        #     _, first_occurrences = np.unique(seq.numpy(), return_index=True)
        #     print(first_occurrences[:5])
        #     print
        return torch.tensor([divergence[i][np.unique(seq.numpy(), return_index=True)[1]].mean() for i,seq in enumerate(input_ids)])

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_ref_probs(dataloader: DataLoader, tokenizer: AutoTokenizer, smoothing: str="laplace") -> Integer[torch.Tensor, "vocab"]:
    """
    Computes reference probabilities of vocab in given token dataloader (Laplace Smoothed)

    Args:
        dataloader: input data to compute vocab frequency over
        tokenizer: tokenizer which contains vocabulary
        smoothing: smoothing to apply ('laplace' or None), defaults to laplace
    Returns:
        Tensor of vocab counts
    """
    freq_counts = torch.zeros(len(tokenizer), dtype=torch.int64)
    for batch in dataloader:
        input_ids = batch["input_ids"] if isinstance(batch, dict) else batch
        for token_seq in input_ids:
            for token in token_seq:
                freq_counts[token]+=1
            # freq_counts += torch.bincount(token_seq, minlength=len(tokenizer))
    if smoothing=="laplace":
        return (freq_counts+1)/(torch.sum(freq_counts)+len(tokenizer))
    elif smoothing is None:
        return freq_counts/torch.sum(freq_counts)
    else:
        raise ValueError(f"Expect smoothing to be 'laplace' or None (got {smoothing}).")