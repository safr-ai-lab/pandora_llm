from tqdm import tqdm
from typing import Union
from jaxtyping import Integer, Float, Bool
import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler

####################################################################################################
# MAIN CLASS
####################################################################################################
class LOSS(FeatureComputer, LLMHandler):
    """
    Computes the negative log-likelihood (NLL) for a given dataset using a pre-trained language model.
    Under strong assumptions, thresholding this is approximately optimal by the Neyman-Pearson lemma:
    MALT from Sablayrolles et al. 2019 (https://arxiv.org/pdf/1908.11229).

    Attributes:
        model (AutoModelForCausalLM): The pre-trained language model to compute the NLL.
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

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

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_log_probs_dl(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    mode: str="mean"
) -> Union[
    Float[torch.Tensor, "n"],
    Float[torch.Tensor, "n seq-1"],
    Float[torch.Tensor, "n seq-1 vocab"],
    Float[torch.Tensor, "n seq-1 vocab+1"],
    Float[torch.Tensor, "n seq-1 3"],
]:
    """
    Computes log probabilities for sequences in a dataloader.

    This function processes a dataloader using a language model and an `Accelerator` for efficiency.
    It can return various forms of log probabilities based on the specified `mode`.

    Args:
        model: The pre-trained causal language model.
        dataloader: The dataloader containing input sequences and optional attention masks.
        accelerator: The `Accelerator` object to manage model and data parallelism.
        mode: Specifies the type of output:
            - "mean": Returns the mean of log probabilities (log-likelihood).
            - "tokens": Returns log probabilities for each token in the sequence.
            - "all": Returns log probabilities for the entire vocabulary for each token.
            - "tokens+all": Returns target log probs stacked on the 0th index of the vocab dimension
            - "tokens+z": Returns target log probs stacked with mean and std of all vocab tokens as the last dimension
            Default is "mean".

    Returns:
        The computed log probabilities based on the selected mode.
    """
    model, dataloader = accelerator.prepare(model, dataloader)
    results = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].detach() if isinstance(batch, dict) else batch.detach()
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        result = compute_log_probs(model, input_ids, attention_mask, mode=mode)
        result = accelerator.gather_for_metrics(result).cpu()
        results.append(result)
    return torch.cat(results,dim=0)

def compute_log_probs(
    model: AutoModelForCausalLM, 
    input_ids: Integer[torch.Tensor, "batch seq"], 
    attention_mask: Bool[torch.Tensor, "batch seq"] = None, 
    mode: str="mean",
) -> Union[
    Float[torch.Tensor, "batch"],
    Float[torch.Tensor, "batch seq-1"],
    Float[torch.Tensor, "batch seq-1 vocab"],
    Float[torch.Tensor, "batch seq-1 vocab+1"],
    Float[torch.Tensor, "batch seq-1 3"],
]:
    """
    Computes log probabilities for a batch of input sequences.

    This function calculates the log probabilities of target tokens in a batch using a causal language model.
    It supports multiple output modes for different use cases.

    Args:
        model: The pre-trained causal language model.
        input_ids: Input token IDs of shape (batch, sequence length).
        attention_mask: Attention mask to ignore padding tokens. If None, a default mask will be generated from `input_ids>0`.
        mode: Specifies the type of output:
            - "mean": Returns the mean of log probabilities (log-likelihood) per sequence.
            - "tokens": Returns log probabilities for each token in the sequence.
            - "all": Returns log probabilities for all vocabulary tokens at each sequence position.
            - "tokens+all": Returns target log probs stacked on the 0th index of the vocab dimension
            - "tokens+z": Returns target log probs stacked with mean and std of all vocab tokens as the last dimension
            Default is "mean".

    Returns:
        The computed log probabilities based on the selected mode.

    Raises:
        ValueError: If the `mode` is not one of "mean", "tokens", or "all".
    """
    model.eval()
    with torch.no_grad():
        if attention_mask is None:
            attention_mask = (input_ids>0).detach()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits: Float[torch.Tensor, "batch seq vocab"] = outputs.logits.detach()
        logits = logits[:, :-1, :].contiguous()
        target = input_ids[:, 1:].contiguous()
        log_probs: Float[torch.Tensor, "batch seq-1 vocab"] = log_softmax(logits, dim=-1)
        if mode == "all":
            return log_probs
        target_log_probs: Float[torch.Tensor, "batch seq-1"] = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if mode=="tokens":
            return target_log_probs
        elif mode=="tokens+all":
            log_probs_with_target: Float[torch.Tensor, "batch seq-1 vocab+1"] = torch.cat((target_log_probs.unsqueeze(-1),log_probs),dim=-1)
            return log_probs_with_target
        elif mode=="tokens+z":
            std_vocab_log_probs, mean_vocab_log_probs = torch.std_mean(log_probs,dim=-1)
            log_probs_with_z: Float[torch.Tensor, "batch seq-1 3"] = torch.stack([target_log_probs,mean_vocab_log_probs,std_vocab_log_probs],dim=-1)
            return log_probs_with_z
        elif mode == "mean":
            target_log_probs = target_log_probs * attention_mask[:, 1:]
            mean_log_probs: Float[torch.Tensor, "batch"] = target_log_probs.sum(dim=1)/(attention_mask[:, 1:].sum(dim=1))
            return mean_log_probs
        else:
            raise ValueError(f"Expect mode to be one of 'mean', 'tokens', 'all', or 'tokens+all' (got {mode})")