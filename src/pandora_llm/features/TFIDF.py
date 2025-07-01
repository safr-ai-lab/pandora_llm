import math
from jaxtyping import Num, Int
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base import FeatureComputer
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF(FeatureComputer):
    """
    TFIDF features

    Attributes:
        vectorizer: CountVectorizer object
        idf (Float[torch.Tensor, "vocab"]): inverse document frequencies for each vocab item
    """
    def __init__(self):
        FeatureComputer.__init__(self)
        self.vectorizer = None
        self.idf = None
    
    def compute_features(self, dataloader: DataLoader, mode: str) -> Num[torch.Tensor, "n vocab"]:
        """
        Computes tfodf features, with mode specifying the type of data the dataloader is holding.

        Args:
            dataloader: dataloader to compute TFIDF features over
            mode: whether to compute bow on 'tokens' or 'text'
        
        Returns:
            TFIDF features for each sample

        Raises:
            ValueError: if mode is not one of 'tokens' or 'text'
        """
        if mode=="tokens":
            return self.compute_tfidf_tokens(dataloader)
        elif mode=="text":
            return self.compute_tfidf_text(dataloader)
        else:
            raise ValueError(f"Mode should be one of 'tokens' or 'text' (got {mode}).")

    def train_tfidf_text(self, dataloader: DataLoader) -> None:
        """
        Trains TFIDF's vocabulary on the given dataloader

        Args:
            dataloader: text dataloader to compute bow features over
        """
        self.vectorizer = TfidfVectorizer()
        self.vectorizer = self.vectorizer.fit(dataloader.dataset)

    def compute_tfidf_text(self, dataloader: DataLoader) -> Num[torch.Tensor, "n vocab"]:
        """
        Computes the tfidf features on the dataset

        Args:
            dataloader: input dataloader to compute tfidf features over
        Returns:
            Bow features for each sample (N x vocab)
        Raises:
            Exception: if did not call train beforehand
        """
        return torch.from_numpy(self.vectorizer.transform(dataloader.dataset).toarray())
    
    def train_tfidf_tokens(self, dataloader) -> None:
        """
        Trains TFIDF's vocabulary on the given dataloader

        Args:
            dataloader: token dataloader to compute bow features over
        """
        doc_freq = {}    
        num_docs = 0
        for i,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            if isinstance(batch, dict):
                batch = batch["input_ids"]
            for doc in batch:
                num_docs += 1
                unique_tokens = torch.unique(doc)
                for token in unique_tokens.tolist():
                    if token not in doc_freq:
                        doc_freq[token] = 0
                    doc_freq[token] += 1
        self.vectorizer = {token: idx for idx, token in enumerate(doc_freq.keys())}
        self.idf = torch.zeros(len(doc_freq))
        for token,idx in self.vectorizer.items():
            self.idf[idx] = math.log((num_docs + 1) / (doc_freq[token] + 1)) + 1

    def compute_tfidf_tokens(self, dataloader: DataLoader) -> Num[torch.Tensor, "n vocab"]:
        """
        Computes the tfidf features on the dataset

        Args:
            dataloader: input dataloader to compute tfidf features over
        Returns:
            Bow features for each sample (N x vocab)
        Raises:
            Exception: if did not call train beforehand
        """
        tfidf_features = []
        for i,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            if isinstance(batch, dict):
                batch = batch["input_ids"]
            tf = torch.zeros(batch.size(0), len(self.vectorizer))

            for doc_idx, doc in enumerate(batch):
                doc_len = len(doc)
                for token in doc.tolist():
                    if token in self.vectorizer:
                        token_idx = self.vectorizer[token]
                        tf[doc_idx, token_idx] += 1
                tf[doc_idx] /= doc_len

            tfidf_batch = tf * self.idf
            tfidf_features.append(tfidf_batch)

        return torch.cat(tfidf_features, dim=0)