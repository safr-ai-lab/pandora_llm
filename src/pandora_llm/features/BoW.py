from jaxtyping import Num, Integer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from .base import FeatureComputer

class BoW(FeatureComputer):
    """
    Bag of words features

    Attributes:
        vectorizer: CountVectorizer object
    """
    def __init__(self):
        FeatureComputer.__init__(self)
        self.vectorizer = None
    
    def compute_features(self, dataloader: DataLoader, mode: str) -> Num[torch.Tensor, "n vocab"]:
        """
        Computes bow features, with mode specifying the type of data the dataloader is holding.

        Args:
            dataloader: dataloader to compute bow features over
            mode: whether to compute bow on 'tokens' or 'text'
        
        Returns:
            BoW features for each sample

        Raises:
            ValueError: if mode is not one of 'tokens' or 'text'
        """
        if mode=="tokens":
            return self.compute_bow_tokens(dataloader)
        elif mode=="text":
            return self.compute_bow_text(dataloader)
        else:
            raise ValueError(f"Mode should be one of 'tokens' or 'text' (got {mode}).")

    def train_bow_text(self, dataloader: DataLoader) -> None:
        """
        Trains BoW's vocabulary on the given dataloader

        Args:
            dataloader: text dataloader to compute bow features over
        """
        self.vectorizer = CountVectorizer()
        self.vectorizer = self.vectorizer.fit(dataloader.dataset)

    def compute_bow_text(self, dataloader: DataLoader) -> Integer[torch.Tensor, "n vocab"]:
        """
        Computes the bow features on the dataloader

        Args:
            dataloader: input dataloader to compute bow features over
        Returns:
            Bow features for each sample (N x vocab)
        Raises:
            Exception: if did not call train beforehand
        """
        if self.vectorizer is None:
            raise Exception("Please train before computing!")
        return torch.from_numpy(self.vectorizer.transform(dataloader.dataset).toarray())
    
    def train_bow_tokens(self, dataloader: DataLoader) -> None:
        """
        Trains TFIDF's vocabulary on the given dataloader

        Args:
            dataloader: token dataloader to compute bow features over
        """
        vocab = set()
        for i,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            if type(batch) == 'dict' or type(batch) == transformers.tokenization_utils_base.BatchEncoding:
                vocab.update(torch.unique(batch["input_ids"]).tolist())
            else:
                vocab.update(np.unique(batch)) # assumes it is a list
        self.vectorizer = {word: idx for idx, word in enumerate(vocab)}

    def compute_bow_tokens(self, dataloader: DataLoader) -> Integer[torch.Tensor, "n vocab"]:
        """
        Computes the bow features on the dataloader

        Args:
            dataloader: input dataloader to compute bow features over
        Returns:
            Bow features for each sample (N x vocab)
        Raises:
            Exception: if did not call train beforehand
        """
        if self.vectorizer is None:
            raise Exception("Please train before computing!")
        bow_features = []
        for i,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            if type(batch) == 'dict' or type(batch) == transformers.tokenization_utils_base.BatchEncoding:
                for seq in batch["input_ids"]:
                    count_vector = torch.zeros(len(self.vectorizer), dtype=torch.long)
                    for token in seq:
                        token = token.item()
                        if token in self.vectorizer:
                            count_vector[self.vectorizer[token]] += 1
                    bow_features.append(count_vector)
            else:
                for seq in batch:
                    count_vector = torch.zeros(len(self.vectorizer), dtype=torch.long)
                    for token in seq:
                        token = token.item()
                        if token in self.vectorizer:
                            count_vector[self.vectorizer[token]] += 1
                    bow_features.append(count_vector)
        return torch.stack(bow_features, dim=0)