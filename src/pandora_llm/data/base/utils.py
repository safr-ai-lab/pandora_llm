from itertools import chain
from transformers import AutoTokenizer
from .Dataset import DatasetWithMetadata, load_dataset_with_metadata

class ChainWithLength:
    def __init__(self, *iterables):
        self.iterables = iterables
        self.total_length = sum(len(it) for it in iterables if hasattr(it, '__len__'))
        self.chain_iter = chain(*iterables)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chain_iter)

    def __len__(self):
        return self.total_length

def collate_fn(batch, tokenizer: AutoTokenizer, **kwargs):
    """
    Apply tokenizer to all elements of batch

    Args:
        batch (list of str): batch of texts to tokenize
        tokenizer (AutoTokenizer): the tokenizer
        kwargs: tokenizer kwargs

    Returns:
        dict: the tokenized results

    """
    return tokenizer(batch, return_tensors="pt", **kwargs)