import os
from jaxtyping import Float, Integer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl
from ..utils.log_utils import clean_filename, mem_stats

####################################################################################################
# MAIN CLASS
####################################################################################################
class Quantile(FeatureComputer,LLMHandler):
    """
    Computes the quantile score from Zhang et al. 2024 (https://arxiv.org/pdf/2409.14513).
    It trains a weak ensemble to estimate the mean and stddev conditional on nonmember data using a "gaussian+pinball" loss.
    The reference is then mean+normal_cdf^{-1}(1-alpha)*stddev.
    Currently supports GPTNeoX models as reference.
    
    """
    def __init__(self, model_name: str, ref_model_name: str, model_revision: str=None, model_cache_dir: str=None, ref_model_revision: str=None, ref_model_cache_dir: str=None):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, model_name, model_revision=model_revision, model_cache_dir=model_cache_dir)
        self.ref_model_name = ref_model_name
        self.ref_model_revision = ref_model_revision
        self.ref_model_cache_dir = ref_model_cache_dir
        self.ref_model_paths = []

    def load_model(self, model_index: int) -> None:
        """
        Loads the specified model into memory.

        Args:
            model_index: Index of the model to load. Base model corresponds to index 0; quantile regression models are 1-indexed.

        Raises:
            IndexError: If the `model_index` is out of bounds.
            Exception: If a model is already loaded.
        """
        if not 0<=model_index<=len(self.ref_model_paths):
            raise IndexError(f"Model index {model_index} out of bounds; should be in [0,{len(self.ref_model_paths)}].")
        if self.model is None:
            if model_index==0:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            else:
                self.model = GPTNeoXForQuantileRegression.from_pretrained(self.ref_model_paths[model_index-1])
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def finetune_ref(self, nonmember_train_tokens: Integer[torch.Tensor, "n seq"], neg_log_probs: Float[torch.Tensor, "n"], accelerator: Accelerator) -> None:
        """
        Fine-tunes reference model ensemble to do quantile regression

        Args:
            nonmember_train_tokens: nonmember train tokens
            neg_log_probs: negative log probs of nonmember train tokens
            accelerator: accelerator object
        """
        if self.model is not None:
            raise Exception("There is already a model in memory; please call .unload_model() first!")

        dataset = Dataset.from_dict({"input_ids":nonmember_train_tokens, "labels":neg_log_probs})
        bootstrapped_indices = torch.randint(0, len(dataset), (len(dataset),))
        dataset = dataset.select(bootstrapped_indices.tolist()).train_test_split(0.1)

        self.model = GPTNeoXForQuantileRegression.from_pretrained(self.ref_model_name, revision=self.ref_model_revision, cache_dir=self.ref_model_cache_dir)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        training_args = TrainingArguments(
            output_dir=os.path.join("models","quantile",clean_filename(f"{self.model_name}-{self.ref_model_name}-{len(self.ref_model_paths)}")),
            eval_strategy="epoch",
            do_train=True,
            num_train_epochs=1,
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            compute_metrics=compute_metrics,
        )
        trainer = accelerator.prepare(trainer)
        trainer.train()
        self.ref_model_paths.append(os.path.join("models","quantile",clean_filename(f"{self.model_name}-{self.ref_model_name}-{len(self.ref_model_paths)}")))

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator, mode: str="primary") -> Float[torch.Tensor, "n"]:
        """
        Computes the negative log-likelihood (NLL) feature for the given dataloader.

        Args:
            dataloader: The dataloader providing input sequences.
            accelerator: The `Accelerator` object for distributed or mixed-precision training.
            mode: whether to compute the "primary" score or the "ref" score

        Returns:
            The NLL feature for each sequence in the dataloader.
        
        Raises:
            Exception: If the model is not loaded before calling this method.
            ValueError: If the model stage is unexpected
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if mode=="primary":
            return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="mean")
        elif mode=="ref":
            return compute_quantile_regression_dl(model=self.model,dataloader=dataloader,accelerator=accelerator)
        else:
            raise ValueError(f"Mode should be 'primary' or 'ref' (got {mode}.")

    @staticmethod
    def reduce(primary_log_probs: Float[torch.Tensor, "n"], ref_mu_sigma: Float[torch.Tensor, "n num_models 2"]) -> Float[torch.Tensor, "n"]:
        """
        Computes quantile attack by computing primary_log_probs minus reference.
        The reference is mean+normal_cdf^{-1}(1-alpha)*stddev.

        Args:
            primary_log_probs: Log probs from primary model
            ref_mu_sigma: Nonmember mu and sigma from reference model
        Returns:
            primary minus ref
        """
        ref_mu = ref_mu_sigma[:,:,0].mean(dim=1)
        ref_sigma = torch.sqrt((ref_mu_sigma[:,:,1]**2+ref_mu_sigma[:,:,0]**2-ref_mu.unsqueeze(-1)**2).mean(dim=1))
        ref_threshold = ref_mu+scipy.stats.norm.ppf(1-0.05)*ref_sigma
        return primary_log_probs-ref_threshold

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

# From https://github.com/amazon-science/llm-qmia

from typing import Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import EvalPrediction
import scipy.stats


def gaussian_loss_fn(score, target, eps=1e-4, quantile=None, kl_weight=0.0, return_kl=False, ignore_index=-100):
    mask = (target == ignore_index)
    # little different from the rest, score is Nx2, quantile is ignored, this is just a negative log likelihood of a Gaussian distribution
    assert (
        score.ndim == 2 and score.shape[-1] == 2
    ), "score has the wrong shape, expected Nx2 input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)
    mu = score[:, 0]
    var = score[:, 1]

    assert (
        mu.shape == var.shape and mu.shape == target.shape
    ), "mean, std and target have non-compatible shapes, got {} {} {}".format(
        mu.shape, var.shape, target.shape
    )

    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    loss = 0.5 * torch.log(var) + 0.5 * (target - mu) ** 2 / (var) + 0.5 * math.log(2 * math.pi)
    assert target.shape == loss.shape, "loss should be a 1-d vector got {}".format(
        loss.shape
    )
    
    kl = 0.5 * (-torch.log(var) + var + mu ** 2 - 1)
    
    if kl_weight > 0:
        loss += kl_weight * kl
    
    if return_kl:
        return loss, kl

    return loss[~mask]

def pinball_loss_fn(score, target, quantiles, ignore_index=-100):
    mask = (target == ignore_index)

    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)

    target = target.reshape([-1, 1])
    delta_score = target - score
    loss = torch.maximum(delta_score * quantiles, delta_score * (quantiles - 1.0))
    return loss[~mask, :]

def gaussian_pinball_loss_fn(score, target, eps=1e-4, quantile=None, ignore_index=-100):
    # little different from the rest, score is Nx2, quantile is ignored, this is just a negative log likelihood of a Gaussian distribution
    assert (
        score.ndim == 2 and score.shape[-1] == 2
    ), "score has the wrong shape, expected Nx2 input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)

    gaussian_loss = gaussian_loss_fn(score, target, ignore_index=ignore_index)
    pinball_loss = pinball_loss_fn(score[:, [0]], target, torch.FloatTensor([0.5]).to(score.device), ignore_index=ignore_index).sum(-1) + pinball_loss_fn(score[:, [0]] + torch.sqrt(score[:, [1]]), target, torch.FloatTensor([1-scipy.stats.norm.sf(1)]).to(score.device), ignore_index=ignore_index).sum(-1)

    loss = gaussian_loss + pinball_loss

    return loss

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    pred_mu = preds[:, 0]
    mse_uncertainty, kl = gaussian_loss_fn(torch.tensor(preds), torch.tensor(p.label_ids), return_kl=True)
    mse_uncertainty = mse_uncertainty.mean().detach().cpu()
    kl = kl.mean().detach().cpu()
    pinball_loss_mean = pinball_loss_fn(torch.tensor(preds[:, [0]]), torch.tensor(p.label_ids), quantiles=torch.tensor([0.5])).mean().detach().cpu()
    pinball_loss_meanpstd = pinball_loss_fn(torch.tensor(preds[:, [0]]) + torch.sqrt(torch.tensor(preds[:, [1]])), torch.tensor(p.label_ids), quantiles=torch.tensor([1-scipy.stats.norm.sf(1)])).mean().detach().cpu()
    mse = torch.nn.functional.mse_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
    mae = torch.nn.functional.l1_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
    result = {
        "mse_uncertainty": mse_uncertainty,
        "mse": mse,
        "mae": mae,
        "kl": kl,
        "pinball_0.5": pinball_loss_mean,
        f"pinball_loss_{1-scipy.stats.norm.sf(1)}": pinball_loss_meanpstd,
    }
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result    

class GPTNeoXForQuantileRegression(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.var_nonlin = torch.nn.functional.softplus
        self.config = config
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                print(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        pooled_logits[:, 1] = self.var_nonlin(pooled_logits[:, 1].clone())

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            loss_fct = gaussian_pinball_loss_fn
            loss = loss_fct(pooled_logits, labels).mean()

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def compute_quantile_regression_dl(model: GPTNeoXForQuantileRegression, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n 2"]:
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()
    with torch.no_grad():
        results = []
        for i,batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].detach() if isinstance(batch, dict) else batch.detach()
            attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
            result = model(input_ids=input_ids, attention_mask=attention_mask).logits
            result = accelerator.gather_for_metrics(result).cpu()
            results.append(result)
        return torch.cat(results,dim=0)