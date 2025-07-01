from jaxtyping import Num, Float
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler
from .LOSS import compute_log_probs_dl

####################################################################################################
# MAIN CLASS
####################################################################################################
class MoPe(FeatureComputer,LLMHandler):
    """
    Model Perturbation (MoPe) Attack

    The MoPe attack explores the effect of small perturbations to a language model's parameters on the loss.
    Introduced by Li et al. 2023 (https://arxiv.org/pdf/2310.14369).

    Attributes:
        new_model_paths (list[str]): Paths to the generated perturbed models.
    """
    def __init__(self,*args,**kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)
        self.new_model_paths = []
        
    def load_model(self, model_index: int) -> None:
        """
        Loads the specified model into memory.

        Args:
            model_index: Index of the model to load. Base model corresponds to index 0; perturbed models are 1-indexed.

        Raises:
            IndexError: If the `model_index` is out of bounds.
            Exception: If a model is already loaded.
        """
        if not 0<=model_index<=len(self.new_model_paths):
            raise IndexError(f"Model index {model_index} out of bounds; should be in [0,{len(self.new_model_paths)}].")
        if self.model is None:
            if model_index==0:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.new_model_paths[model_index-1])
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def generate_new_models(self, tokenizer: AutoTokenizer, num_models: int, noise_stdev: float, noise_type: str="gaussian") -> None:
        """
        Generates perturbed versions of the base model and saves them to disk.

        Args:
            tokenizer: Tokenizer associated with the base model.
            num_models: Number of perturbed models to generate.
            noise_stdev: Standard deviation of the noise added to model parameters (or scale for Rademacher noise).
            noise_type: Type of noise to apply: 'gaussian' or 'rademacher'. Defaults to 'gaussian'.

        Raises:
            ValueError: If an unsupported `noise_type` is specified.
        """
        self.new_model_paths = []

        with torch.no_grad():
            for model_index in range(1, num_models+1):  
                print(f"Loading Perturbed Model {model_index}/{num_models}")      

                dummy_model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)

                # Perturb model
                for name, param in dummy_model.named_parameters():
                    if noise_type == 'gaussian':
                        noise = torch.randn(param.size()) * noise_stdev
                    elif noise_type == 'rademacher':
                        noise = (torch.randint(0,2,param.size())*2-1) * noise_stdev
                    else:
                        raise ValueError(f"Noise type not recognized: {noise_type}")
                    param.add_(noise)
                
                # Move to disk 
                dummy_model.save_pretrained(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}", from_pt=True)
                tokenizer.save_pretrained(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}")
                self.new_model_paths.append(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}")
                del dummy_model, name, param
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def compute_features(self, dataloader: DataLoader, accelerator: Accelerator) -> Num[torch.Tensor, "n ..."]:
        """
        Compute the LOSS statistic for a dataloader using the currently loaded model.

        Args:
            dataloader: input data to compute statistic over
            accelerator: accelerator object
        Returns:
            Loss statistic
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="mean")
    
    @staticmethod
    def reduce(log_probs: Float[torch.Tensor, "model n"]) -> Float[torch.Tensor, "n"]:
        """
        Computes the difference in log probs between perturbed and unperturbed models.
        Assumes log_probs[0] is the log probs for the unperturbed model.

        Args:
            log_probs: M x n tensor of log probs where M is the number of models and n is the number of samples
        Returns:
            MoPe statistic
        """
        return -(log_probs[1:,:].mean(dim=0)-log_probs[0,:])

