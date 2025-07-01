import math
from itertools import groupby
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from .base.Dataset import DatasetWithMetadata, DatasetDictWithMetadata

"""
Query data. We can do {packed val, unpacked val} x 
{
   packed train dupe, 
   unpacked train dupe, 
   packed train deduped,
}
Filtered by min_length and max_length (document and window-wise, for unpacked and packed, respectively)

Turn any combination of this into:
DatasetDictWithMetadata({
    member: DatasetWithMetadata({
        features: ['tokens', 'text'],
        num_rows: X,
	… other metadata … 
    })
    nonmember: DatasetWithMetadata({
        features: ['tokens', 'text'],
        num_rows: X,
	… other metadata … 
    })
})
"""

class ThePile:
    @classmethod
    def load_data(cls, number=2000, start_index=0, seed=229, valpack=True, trainpack=True, traindupe=True, valpack_eos=False):
        """
        Loads train and val data per user flags from The Pile (raw), and organizes into DatasetDictWithMetadata.
        """
        if valpack ^ trainpack:
            raise ValueError("Train and val data should both be packed, or unpacked.")
        if not (trainpack or traindupe):
            raise ValueError("The pythia-deduped models do not have EOS tokens in their training data.")
        
        # Load train data
        train_dataset = ThePile.load_train(number=number, start_index=start_index, seed=seed, deduped=(not traindupe), unpack=(not trainpack))

        # Load val data
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
        val_dataset = ThePile.load_val(number=number, start_index=start_index, seed=seed, tokenizer=tokenizer, window=2048 if valpack else 0, pack_eos=valpack_eos)

        return DatasetDictWithMetadata({
            "member": train_dataset,
            "nonmember": val_dataset,
        })
    
    @classmethod
    def load_train(cls, number=1000, percentage=None, start_index=0, seed=229, deduped=True, unpack=False, min_length=20):
        """
        Load train pile samples from random deduped sampler.

        NOTE: min_length is only supported during unpacking

        Args:
            number (int): Number of samples
            percentage (float): Percentage of total samples (if number not specified). Default to None
            start_index (int): What index to start counting samples from. Default to 0
            seed (int): Random seed
            num_splits (int): Number of splits to separate data into. Usually set to 1 in most methods
            deduped (bool): Whether to load the deduped (True) or duped (False) Pile
            unpack (bool): Unpacks samples 
            min_length (int): Filter out sequences that are below this minimum number of tokens 
        Returns:
            list[list[str]]: List of splits, each split is a list of text samples.
        """
        if deduped and unpack:
            raise NotImplementedError("Deduped pile train random sampler does not have EOS tokens, so cannot unpack.")
        if deduped:
            dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled",split="train",revision="d2c194e").shuffle(seed=seed)
        else:
            dataset = load_dataset("EleutherAI/pile-duped-pythia-random-sampled",split="train",revision="a7b374f").shuffle(seed=seed)
        clip_len = number if percentage is None else int(len(dataset)*percentage)
        if not (1<=clip_len<=len(dataset)):
            raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
        
        def unpack_fn(examples):
            chunks = []
            for sample in examples:
                result = []
                for k, group in groupby(sample, lambda y: y == tokenizer.eos_token_id):
                    input_ids= list(group)
                    if (not k) and len(input_ids)>min_length:
                        result.append(tokenizer.decode(input_ids))
                chunks.extend(result)
            return chunks

        dataset = dataset.select(range(start_index,start_index+clip_len))
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

        if unpack:
            clip_len = number if percentage is None else int(len(dataset)*percentage)
            dataset = dataset.map(lambda x: {"text": unpack_fn(x["tokens"])},remove_columns=["index","is_memorized"],batched=True).shuffle(seed=seed)
            dataset = dataset.select(range(clip_len))
        else:
            dataset = dataset.map(lambda x: {"text": tokenizer.decode(x["tokens"])},remove_columns=["index","is_memorized"])
        
        return DatasetWithMetadata(
            dataset,
            name="pile",
            split="train",
            number=number,
            percentage=percentage,
            start_index=start_index,
            seed=seed,
            deduped=deduped,
            unpack=unpack,
            min_length=min_length,
        )

    @classmethod
    def load_val(cls, number=1000, percentage=None, start_index=0, seed=229, tokenizer=None, window=2048, compensation_factor=2., uncopyright=False, pack_eos=False):
        """
        Loads the validation pile (NOT deduped), does an exact match deduplication and shuffling, 
        packs samples into 2048-sized chunks, and returns the specified number of splits. 

        Args:
            number (int): Number of samples
            percentage (float): Percentage of total samples (if number not specified). Default to None
            start_index (int): What index to start counting samples from. Default to 0
            seed (int): Random seed
            num_splits (int): Number of splits to separate data into. Usually set to 1 in most methods
            window (int): number of tokens to pack up to. If not packing, set to 0.
            compensation_factor (float): when packing, sample this times more samples to compensate for packing. Default to 2.
            uncopyright (bool): whether to load uncopyrighted version. Default to False
            pack_eos (bool): whether to add an eos token in between packed samples. Default to False
        
        Returns:
            list[list[str]]: List splits, each split is a list of text samples.
        """
        if uncopyright:
            dataset = load_dataset("the_pile_val.py", split="validation").shuffle(seed=seed)
        else:
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
        clip_len = number if percentage is None else int(len(dataset)*percentage)

        if window==0: # No packing
            dataset = dataset.select(range(start_index,start_index+clip_len))
            # dataset = list(dict.fromkeys(entry["text"] for entry in dataset))[:clip_len]
        else:
            # Use a multiple of clip_len to ensure enough samples after packing
            if not (1<=int(clip_len*compensation_factor)<=len(dataset)):
                raise IndexError(f"Number or percentage out of bounds. You specified {int(clip_len*compensation_factor)} samples but there are only {len(dataset)} samples.")
            if tokenizer is None:
                raise ValueError("Must specify tokenizer to pack.")
            dataset = dataset.select(range(start_index,start_index+int(clip_len*compensation_factor)))
            dataset = dataset.map(lambda x: {"tokens": tokenizer(x["text"])["input_ids"]}, remove_columns=["meta"])

            # Get tokens for everything, and add EOS_token between examples
            collated_docs_with_eos_split = []
            if pack_eos:
                print("==================================================================")
                print("WARNING: PACKING VAL PILE WITH EOS TOKENS IS INCORRECT BEHAVIOR!!!\n"*3)
                print("==================================================================")
            for item in tqdm(dataset["tokens"]):
                if pack_eos:
                    collated_docs_with_eos_split += item + [tokenizer.eos_token_id]
                else:
                    collated_docs_with_eos_split += item

            # Turn tokens back into strings. 
            packed_dataset = {"tokens": [], "text": []}
            for i in tqdm(range(int(math.ceil(len(collated_docs_with_eos_split) / window)))):
                packed_segment = collated_docs_with_eos_split[window * i:window * (i+1)]
                if len(packed_segment)!=window:
                    packed_segment.extend([0 for _ in range(window-len(packed_segment))])
                packed_dataset["tokens"].append(packed_segment)
                packed_dataset["text"].append(tokenizer.decode(packed_segment))
            packed_dataset["tokens"] = np.array(packed_dataset["tokens"],dtype=np.uint16)
            dataset = Dataset.from_dict(packed_dataset)
            if len(dataset)>clip_len:
                dataset = dataset.select(range(clip_len))
            else:
                print("WARNING: Packing resulted in less samples than expected!!!")
        
        return DatasetWithMetadata(
            dataset,
            name="pile",
            split="val",
            number=number,
            percentage=percentage,
            start_index=start_index,
            seed=seed,
            window=window,
            compensation_factor=compensation_factor,
            uncopyright=uncopyright,
            pack_eos=pack_eos,
        )