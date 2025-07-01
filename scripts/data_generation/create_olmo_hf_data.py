"""
Spec:
- Creates Olmo data with split at 400_000 batches
"""


import random
from pandora_llm.data.create_olmo_data import create_olmo_data
from pandora_llm.data import DatasetDictWithMetadata, DatasetWithMetadata, load_dataset_with_metadata
import numpy as np
from datasets import Dataset# , DatasetDict
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
import os

def npArrayToDataset(nparray, tokenizer):
    """
    Token processing and text conversion for nparray of token.
    """
    # Tokens
    tokens = nparray.tolist()

    # Text
    strings_list = []
    for idx, token_ids in enumerate(tqdm(nparray, desc="Converting tokens to text")):
        # Convert token IDs to a list of integers
        token_ids_list = token_ids.tolist()
        # Decode the token IDs to a string
        decoded_str = tokenizer.decode(token_ids_list)
        strings_list.append(decoded_str)
    
    dataset = Dataset.from_dict({'tokens': tokens, 'text': strings_list})
    return dataset

# Set up args
class Args:
    def __init__(self):
        self.data_order_file_path = "https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy"
        self.train_config_path = "src/pandora_llm/data/OLMo-7B-local.yaml"
        self.model_name = "OLMO7B-local"
        self.start_batchno = 0
        self.batchno = 400000
        self.end_batchno = None
        self.num_data = 100
        self.seed = 229

def main():
    print("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    
    args = Args()    
    # Run data creation
    print("Creating OLMo data...")
    import argparse
    args_namespace = argparse.Namespace(
        data_order_file_path=args.data_order_file_path,
        train_config_path=args.train_config_path,
        model_name=args.model_name,
        start_batchno=args.start_batchno,
        batchno=args.batchno,
        end_batchno=args.end_batchno,
        num_data=args.num_data,
        seed=args.seed
    )
    create_olmo_data(args_namespace)

    print("Loading saved data...")
    # Load the saved data
    train_data = torch.load(f"Data/{args.model_name}_num_data={args.num_data}_start={args.start_batchno}_middle={args.batchno}_end={432410}_bs=2160_train.pt",weights_only=False)
    valid_data = torch.load(f"Data/{args.model_name}_num_data={args.num_data}_start={args.start_batchno}_middle={args.batchno}_end={432410}_bs=2160_valid.pt",weights_only=False)

    # Convert to datasets
    preXdataset = npArrayToDataset(train_data["data"], tokenizer)
    postXdataset = npArrayToDataset(valid_data["data"], tokenizer)

    # Set step info for metadata
    preA = args.start_batchno
    preB = args.batchno
    postA = args.batchno
    postB = args.end_batchno if args.end_batchno else "end"
    data = {'member': preXdataset, 'nonmember': postXdataset}
    args_metadata = {
        'data_order_file_path': args.data_order_file_path,
        'train_config_path': args.train_config_path,
        'model_name': args.model_name,
        'start_batchno': args.start_batchno,
        'batchno': args.batchno,
        'end_batchno': args.end_batchno,
        'num_data': args.num_data,
        'seed': args.seed
    }    
    metadata = {
        'member': {**args_metadata, 'steps': f"{preA}_{preB}"},
        'nonmember': {**args_metadata, 'steps': f"{postA}_{postB}"}
    }
    obj = DatasetDictWithMetadata(datasets=data, metadata=metadata)
    obj.push_to_hub(f"mfli314/OLMo-7B-mia-n{args.num_data}")
    del obj

if __name__ == '__main__':
    main()
