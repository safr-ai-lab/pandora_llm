from jaxtyping import Num, Float
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from .base import FeatureComputer

class Word2Vec(FeatureComputer):
    """
    Word2vec features

    Attributes:
        w2v_model (Word2Vec): Word2vec model
        embedding_dim (int): Word2vec model embedding dimension 
    """
    def __init__(self):
        FeatureComputer.__init__(self)
        self.w2v_model = None
        self.embedding_dim = None

    def load_pretrained_model(self) -> None:
        """
        Loads the pretrained Word2Vec model.

        The pretrained model is downloaded from gensim's dataset repository
        (`word2vec-google-news-300`). After loading, the embedding dimension
        is set to 300.
        """
        self.w2v_model = api.load('word2vec-google-news-300')
        self.embedding_dim = 300

    def compute_features(self, dataloader: DataLoader, mode: str, tokenizer: AutoTokenizer=None) -> Float[torch.Tensor, "n d"]:
        """
        Computes Word2Vec features for the given data in either 'tokens' or 'text' mode.

        Args:
            dataloader: Input data to compute features from.
            mode: Mode of feature computation. Options:
                - "tokens": Compute Word2Vec features for tokenized data.
                - "text": Compute Word2Vec features for full text samples.
            tokenizer: Tokenizer to map token IDs to words (required for 'tokens' mode).

        Returns:
            A tensor of shape (n, d), where n is the number of samples 
            and d is the embedding dimension.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode=="tokens":
            return self.compute_w2v_tokens(dataloader, tokenizer=tokenizer)
        elif mode=="text":
            return self.compute_w2v_text(dataloader)
        else:
            raise ValueError(f"Mode should be one of 'tokens' or 'text' (got {mode}).")

    def compute_w2v_text(self, dataloader: DataLoader) -> Float[torch.Tensor, "n d"]:
        """
        Computes Word2Vec features for text data in the dataloader.

        This method computes the average Word2Vec embedding for each text sample
        by averaging the embeddings of all words in the sample. Words that are
        not in the Word2Vec vocabulary are assigned a zero vector.

        Args:
            dataloader: Dataloader containing text samples.

        Returns:
            A tensor of shape (n, d) containing the average Word2Vec
            embeddings for each text sample.
        """
        tokenized_text = [text.split() for text in dataloader.dataset]
        features = []
        for tokens in tokenized_text:
            embeddings = []
            for word in tokens:
                if word in self.w2v_model:
                    embeddings.append(self.w2v_model[word])
                else:
                    # Handle OOV words with a zero vector or random initialization
                    embeddings.append(np.zeros(self.embedding_dim))
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
            else:
                avg_embedding = np.zeros(self.embedding_dim)
            features.append(avg_embedding)
        return torch.tensor(features, dtype=torch.float)

    def compute_w2v_tokens(self, dataloader: DataLoader, tokenizer: AutoTokenizer) -> Float[torch.Tensor, "n d"]:
        """
        Computes Word2Vec features for tokenized data.

        This method computes the average Word2Vec embedding for each sequence of tokens
        by averaging the embeddings of all valid tokens. Tokens not in the Word2Vec vocabulary
        are assigned a zero vector.

        Args:
            dataloader: Dataloader providing batches of tokenized data.
            tokenizer: Tokenizer used to convert token IDs to words.

        Returns:
            A tensor of shape (n, d) containing the average Word2Vec embeddings for each sequence of tokens.
        """
        features = []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if isinstance(batch, dict):
                batch = batch["input_ids"]
            input_ids = batch
            for seq in input_ids:
                tokens = tokenizer.convert_ids_to_tokens(seq)
                embeddings = []
                for token in tokens:
                    # Clean the token (e.g., remove special tokens)
                    if token in self.w2v_model:
                        embeddings.append(self.w2v_model[token])
                    else:
                        embeddings.append(np.zeros(self.embedding_dim))
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = np.zeros(self.embedding_dim)
                features.append(avg_embedding)
        return torch.tensor(features, dtype=torch.float)