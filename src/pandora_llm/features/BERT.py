from jaxtyping import Num, Int, Float
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AutoTokenizer
from accelerate import Accelerator
from .base import FeatureComputer

class BertFeatureComputer(FeatureComputer):
    """
    BERT features attack

    Attributes:
        bert_model (BertModel): BERT model
        tokenizer (BertTokenizer): BERT tokenizer
        embedding_dim (int): BERT embedding dimension
    """
    def __init__(self, *args, **kwargs):
        FeatureComputer.__init__(self)
        self.bert_model = None
        self.tokenizer = None
        self.embedding_dim = None

    def load_pretrained_model(self) -> None:
        """
        Loads the pretrained BERT model and tokenizer from HuggingFace.

        After loading, the model is set to evaluation mode, and the embedding dimensionality
        is stored in `self.embedding_dim`.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()  # Set the model to evaluation mode
        self.embedding_dim = self.bert_model.config.hidden_size  # Typically 768

    def unload_model(self) -> None:
        """
        Unloads the BERT model and tokenizer from memory.
        """
        self.tokenizer = None
        self.bert_model = None
        self.embedding_dim = None

    def compute_features(self, dataloader: DataLoader, mode: str, accelerator: Accelerator, tokenizer: AutoTokenizer=None) -> Float[torch.Tensor, "n d"]:
        """
        Computes BERT features for the given dataloader.
        Note that the context window of BERT may be smaller than the given input.
        In this case, it truncates.

        Args:
            dataloader: Input data to compute features from.
            mode: Feature computation mode. Either "tokens" or "text".
            accelerator: HuggingFace's Accelerator for efficient computation.
            tokenizer: Tokenizer to map token IDs to words (required for "tokens" mode).

        Returns:
            A tensor of shape (n, d), where n is the number of samples 
            and d is the embedding dimension.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode=="tokens":
            return self.compute_bert_tokens(dataloader, accelerator=accelerator, tokenizer=tokenizer)
        elif mode=="text":
            return self.compute_bert_text(dataloader, accelerator=accelerator)
        else:
            raise ValueError(f"Mode should be one of 'tokens' or 'text' (got {mode}).")        

    def compute_bert_text(self, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n d"]:
        """
        Computes BERT features for full text samples in the dataloader.

        This method computes the [CLS] token embedding for each text sample in batches.

        Note that the context window of BERT may be smaller than the given input.
        In this case, it truncates.

        Args:
            dataloader: Input dataset containing batches of text data.
            accelerator: HuggingFace's Accelerator for efficient computation.

        Returns:
            torch.Tensor: A tensor of shape (n, d) containing the [CLS] embeddings.
        """
        features = []
        self.bert_model, dataloader = accelerator.prepare(self.bert_model, dataloader)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
                inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
                outputs = self.bert_model(**inputs)
                outputs = accelerator.gather_for_metrics(outputs)
                # Get the [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
                features.append(cls_embedding)
        return torch.cat(features, dim=0)
    
    def compute_bert_tokens(self, dataloader: DataLoader, accelerator: Accelerator, tokenizer: AutoTokenizer) -> Float[torch.Tensor, "n d"]:
        """
        Computes BERT features for tokenized data in the dataloader.

        This method computes the [CLS] token embedding for each tokenized sequence.

        Note that the context window of BERT may be smaller than the given input.
        In this case, it truncates.

        Args:
            dataloader: Dataloader providing batches of tokenized data.
            accelerator: HuggingFace's Accelerator for efficient computation.
            tokenizer: Tokenizer to map token IDs to words.

        Returns:
            A tensor of shape (n, d) containing the [CLS] embeddings.
        """
        features = []
        self.bert_model, dataloader = accelerator.prepare(self.bert_model, dataloader)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"] if isinstance(batch, dict) else batch
                text_data = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                retokenized = self.tokenizer(text_data, return_tensors='pt', truncation=True, max_length=512, padding=True)
                retokenized = {key: value.to(accelerator.device) for key, value in retokenized.items()}
                outputs = self.bert_model(**retokenized)
                outputs = accelerator.gather_for_metrics(outputs)
                # Get the [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                features.append(cls_embeddings)
        return torch.cat(features, dim=0)