from typing import Tuple
from jaxtyping import Float, Integer, Bool
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from einops import einsum
from trak.projectors import AbstractProjector, BasicProjector, CudaProjector, ProjectionType
from .base import FeatureComputer, LLMHandler

####################################################################################################
# MAIN CLASS
####################################################################################################
class ModelStealing(FeatureComputer,LLMHandler):
    """
    Model stealing attack
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def prepare_projection(self, 
        svd_dataloader: DataLoader,
        proj_type: str="rademacher",
        proj_dim: int=512,
        proj_seed: int=229,
        device: str=None,
        fp16: bool=False,
    ) -> Tuple[Float[torch.Tensor, "..."],AbstractProjector]:
        """
        Compute the embedding projection layer for the gray-box model-stealing attack

        Args:
            svd_dataloader: input data to estimate projection layer
            proj_type: projection type (defualt "rademacher")
            proj_dim: project to how many dimensions (default 512)
            proj_seed: random seed for random projection
            device: device to compute on: "cuda" or "cpu"
            fp16: whether to use fp16
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
       
        ####################################################################################################
        # OBTAIN PROJECTORS
        ####################################################################################################
        dataloader_logits = compute_dataloader_logits_embedding(model=self.model, dataloader=svd_dataloader, device=device, fp16=fp16).T.float().to(device)
    
        ## Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
        U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
        svd_embedding_projection_layer = U[:,:(next(self.model.parameters()).shape[1])] @ torch.diag(S[:(next(self.model.parameters()).shape[1])])
        
        ## Identify base change to convert regular gradients to gradients we can access
        if device=="cpu":     
            one_sided_grad_project = BasicProjector(50304, proj_dim, proj_seed, ProjectionType(proj_type), 'cpu', 100)
        else:
            one_sided_grad_project = CudaProjector(50304, proj_dim, proj_seed, ProjectionType(proj_type), device, 8)
        return svd_embedding_projection_layer, one_sided_grad_project

    def compute_features(self,
        dataloader: DataLoader,
        svd_embedding_projection_layer: Float[torch.Tensor, "..."],
        projector: AbstractProjector,
        device: str=None,
        fp16: bool=None,
    ) -> Float[torch.Tensor, "n proj_dim"]:
        '''
        Computes dataloader gradients with jl dimensionality reduction.
        Args:
            dataloader: DataLoader of samples.
            svd_embedding_projection_layer: dictionary of dimensionality reduction functions        
            device: CPU or GPU 
            fp16: use half precision floats for model

        Returns:
            torch.Tensor or list: data for input IDs
        
        Raises:
            Exception: must load the model first
        '''
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        return compute_dataloader_basis_changes(model=self.model, dataloader=dataloader, svd_embedding_projection_layer=svd_embedding_projection_layer, projector=projector, device=device, fp16=fp16).cpu()


####################################################################################################
# HELPER FUNCTIONS
####################################################################################################
def compute_dataloader_basis_changes(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    svd_embedding_projection_layer: Float[torch.Tensor, "..."],
    projector: AbstractProjector,
    device: str=None,
    fp16: bool=False,
) -> Float[torch.Tensor, "n proj_dim"]:
    '''
    Computes dataloader gradients with jl dimensionality reduction.
    Args:
        model: HuggingFace model.
        dataloader: DataLoader of samples.
        projector: dictionary of dimensionality reduction functions        
        device: CPU or GPU 
        fp16: use half precision floats for model

    Returns:
        JL-reduced Dataloader gradients for input IDs
    '''
    if fp16:
        model.half()
        svd_embedding_projection_layer.half()
    model.to(device)
    svd_embedding_projection_layer.to(device)

    grads = []
    for data_x in tqdm(dataloader):
        ## Get predictions on data 
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        data_x = data_x.detach()                

        ## Compute features on input data
        grad = compute_basis_change(model=model, svd_embedding_projection_layer=svd_embedding_projection_layer, projector=projector, input_ids=data_x, device=device).detach().cpu()
        grads.append(grad)

    return torch.stack(grads)

def compute_basis_change(
    model: AutoModelForCausalLM,
    svd_embedding_projection_layer: Float[torch.Tensor, "..."],
    projector: AbstractProjector,
    input_ids: Integer[torch.Tensor, "batch seq"],
    attention_mask: Bool[torch.Tensor, "batch seq"]=None,
    device=None
) -> Float[torch.Tensor, "..."]:
    """
    This computes the basis change for the last layer (Carlini et al. gray-box attack), and returns it
    with the norms of that layer.

    Args:
        model: HuggingFace model.
        input_ids: tensor of input IDs.
        svd_embedding_projection_layer: embedding layer
        projector: dimensionality reduction function
        device: CPU or GPU 
    
    Returns:
        Basis change for the last layer
    """
    if attention_mask is None:
        attention_mask  = (input_ids > 0).detach()
    model.eval()
    model.zero_grad()
    outputs = model(input_ids=input_ids.to(device),attention_mask=attention_mask.to(device),labels=input_ids.to(device),output_hidden_states=True)
    outputs.logits.retain_grad()
    outputs.loss.backward()
    
    implied_latents = torch.linalg.lstsq(svd_embedding_projection_layer, outputs.logits[0].T).solution.T
    grad = einsum(outputs.logits.grad, implied_latents, "a b c, b d -> a c d")
    
    ## Norm and projected features
    L = torch.tensor([torch.norm(grad.flatten().view(-1,1).T,p=p) for  p in [float("inf"),1,2]])
    print(type(projector))
    projected = projector.project(grad[0,:,:].T.clone().to(device).contiguous(),1).flatten()

    return torch.concat((L.cpu(),projected.cpu()),dim=0).flatten()


def compute_dataloader_logits_embedding(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: str=None,
    fp16: bool=False
) -> Float[torch.Tensor, "n batch vocab"]:
    '''
    Computes logits of text in dataloader

    Args:
        model: HuggingFace model.
        dataloader: DataLoader with tokens.
        device: CPU or GPU 
        fp16: use half precision floats for model

    Returns:
        Tensor of logits for last token
    '''
    if fp16:
        model.half()
    model.eval()
    model.to(device)

    losses = []
    for data_x in tqdm(dataloader):
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        data_x = data_x.detach()                
        loss = compute_input_ids_logits(model, data_x, device=device).detach().cpu()
        losses.append(loss)        
    
    return torch.stack(losses)

def compute_input_ids_logits(
    model: AutoModelForCausalLM,
    input_ids: Integer[torch.Tensor, "batch seq"],
    attention_mask: Bool[torch.Tensor, "batch seq"]=None,
    device: str=None,
) -> Float[torch.Tensor, "batch vocab"]:
    """
    Compute logits of last token in input ids

    Args:
        model: HuggingFace model.
        input_ids: tensor of input IDs
        attention_mask: attention_mask
        device: CPU or GPU

    Returns:
        Logits of last token in input ids
    """
    with torch.no_grad():
        if attention_mask is None:
            attention_mask  = (input_ids > 0).detach()
        outputs = model(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device),output_hidden_states=True)
        return outputs.logits[0,-1,:]