from typing import Union, Tuple
from jaxtyping import Num, Int, Float
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .base import FeatureComputer, LLMHandler

####################################################################################################
# MAIN CLASS
####################################################################################################
class ALoRa(FeatureComputer,LLMHandler):
    """
    Approximate loss ratio thresholding attack (vs. pre-training)
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        LLMHandler.__init__(self, *args, **kwargs)

    def compute_features(self, 
        dataloader: DataLoader,
        learning_rate: float,
        accelerator: Accelerator,
    ) -> Tuple[Float[torch.Tensor, "n"], Float[torch.Tensor, "n"]]:
        """
        Compute the approximate loss ratio statistic for a given dataloader.
        Assumes batch size of 1.

        Args:
            dataloader: input data to compute statistic over
            learning_rate: learning rate
            accelerator: accelerator object
        Returns:
            Loss after the gradient descent step and loss before the gradient descent step
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if dataloader.batch_size!=1:
            raise Exception("ALoRa is only implemented for batch size 1")
        
        base_statistics = []
        stepped_statistics = []

        self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
            # 1. Forward pass
            if isinstance(data_x, dict):
                data_x = data_x["input_ids"]
            outputs = self.model(data_x, labels=data_x)
            initial_loss = outputs.loss
            accelerator.backward(initial_loss)
            initial_loss = accelerator.gather_for_metrics(initial_loss)

            base_statistics.append(initial_loss.detach().cpu())

            # 2. Perform gradient descent
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-learning_rate)  # Gradient descent step

            # 3. Compute the loss after update
            output_after_ascent = self.model(data_x, labels=data_x)
            new_loss = output_after_ascent.loss
            new_loss = accelerator.gather_for_metrics(new_loss)
            stepped_statistics.append(new_loss.detach().cpu())

            # 4. Restore model to original state
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=learning_rate)  # Reverse gradient descent step
            self.model.zero_grad()  # Reset gradients

        return torch.tensor(stepped_statistics), torch.tensor(base_statistics)

    @staticmethod
    def reduce(stepped_statistics: Float[torch.Tensor, "n"], base_statistics: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n"]:
        return stepped_statistics-base_statistics