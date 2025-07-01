from typing import Iterable, Union
from jaxtyping import Float, Integer, Bool
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from deepspeed.utils import safe_get_full_grad
from .base import FeatureComputer, LLMHandler

####################################################################################################
# MAIN CLASS
####################################################################################################
class GradNorm(FeatureComputer,LLMHandler):
    """
    GradNorm thresholding attack
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def compute_features(self,
        dataloader: DataLoader,
        accelerator: Accelerator,
        norms: Iterable[Union[int,float]]=[1,2,float("inf")],
        gradient_checkpointing: bool=False,
    ) -> dict[str,Float[torch.Tensor, "n layer norm"]]:
        """
        Compute the layerwise gradient norms for a given dataloader, using the specified norms.

        Args:
            dataloader: input data to compute statistic over
            accelerator: accelerator object
            norms: list of norm orders
            gradient_checkpointing: whether to use gradient checkpointing to save memory
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        embedding_layer = self.model.get_input_embeddings().weight
        return compute_gradnorms_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,norms=norms,gradient_checkpointing=gradient_checkpointing)
    
####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_gradnorms_dl(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    norms: Iterable[Union[int,float]]=[1,2,float("inf")],
    gradient_checkpointing: bool=False,
) -> Float[torch.Tensor, "n layer norm"]:
    '''
    Computes gradient norms of dataloader.
    
    Args:
        model: HuggingFace 
        dataloader: DataLoader of samples
        accelerator: Accelerator object
        norms: Gradient norm types

    Returns:
        Gradient norms of each layer, in a tensor of shape [n x layer x norm]
    '''
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model, dataloader = accelerator.prepare(model, dataloader)
    results = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].detach() if isinstance(batch, dict) else batch.detach()
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        grads = compute_gradnorms(model=model, input_ids=input_ids, accelerator=accelerator, norms=norms, attention_mask=attention_mask)
        grads = accelerator.gather_for_metrics(grads).cpu()
        results.append(grads)
    return torch.cat(results,dim=0)

def compute_gradnorms(
    model: AutoModelForCausalLM,
    input_ids: Integer[torch.Tensor, "batch seq"],
    accelerator: Accelerator,
    norms: Iterable[Union[int,float]]=[1,2,float("inf")],
    attention_mask: Bool[torch.Tensor, "batch seq"] = None, 
) -> Float[torch.Tensor, "batch layer norm"]:
    """
    Computes gradient norm of each layer
    Note: takes advantage of the fact that norm([a,b])=norm([norm(a),norm(b)])
    
    Args:
        model: HuggingFace model
        input_ids: tensor of input IDs
        accelerator: Accelerator object
        norms: Gradient norm types
        attention_mask: Attention mask
        
    Returns:
        Gradient norms of each layer, in a tensor of shape [batch x layer x norm]
    """
    if attention_mask is None:
        attention_mask = (input_ids>0).detach()
    model.eval()
    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    accelerator.backward(outputs.loss)
    grad_norms: Float[torch.Tensor, "batch layer norm"] = torch.zeros(input_ids.shape[0],sum(1 for _ in model.named_parameters()),len(norms)).to(accelerator.device)
    for i, (name,param) in enumerate(model.named_parameters()):
        grad = safe_get_full_grad(param).flatten()
        for j, p in enumerate(norms):
            grad_norms[:,i,j] = torch.norm(grad,p=p)    
    return grad_norms
