import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import list_repo_refs
from accelerate.utils import set_seed
from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from .base.TextDataset import DatasetDictWithMetadata

class Dolma:
    @classmethod
    def load_data(
        start_batchno=0,
        val_batchno=400_000,
        end_batchno=None,
        num_data=1_000,
        data_order_file_path="https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy",
        train_config_path="../src/pandora_llm/datasets/OLMo-7B-local.yaml",
        save_to_hf_file=None,
        **kwargs
    ):
        cfg = TrainConfig.load(train_config_path)
        dataset = build_memmap_dataset(cfg, cfg.data)
        batch_size = cfg.global_train_batch_size
        global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

        end_batchno = end_batchno if end_batchno else (len(global_indices)-1) // batch_size
        assert start_batchno < val_batchno < end_batchno, "Need start_batchno < batchno < end_batchno"
        assert batch_size*val_batchno < len(global_indices), "Need batchno * batch_size to be smaller than number of samples in first epoch"

        def get_batch_instances(batch_idx: int, batch_index : int) -> list[int]:
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_indices = global_indices[batch_start:batch_end]
            return dataset[batch_indices[batch_index]]["input_ids"].tolist()

        # Collect training and validation data by sampling from batches w/o replacement    
        train_positions  = np.random.choice(list(range(batch_size*start_batchno,
                                                    batch_size*val_batchno)),
                                            size=num_data,replace=False)
        valid_positions  = np.random.choice(list(range(batch_size*(val_batchno+10),
                                                    batch_size*(end_batchno))),
                                            size=num_data,replace=False)

        train_batches, train_batch_index    = train_positions // batch_size, train_positions % batch_size
        valid_batches, valid_batch_index    = valid_positions // batch_size, valid_positions % batch_size
        
        train_data, valid_data = [], []
        
        print("Collecting training and validation data") 
        for i in tqdm(range(num_data)):
            train_data.append(get_batch_instances(train_batches[i], train_batch_index[i]))
            valid_data.append(get_batch_instances(valid_batches[i], valid_batch_index[i]))

        train_data = {
            "tokens":torch.tensor(train_data),
            "positions":torch.tensor(train_positions)
        }
        valid_data= {
            "tokens":torch.tensor(valid_data),
            "positions":torch.tensor(valid_positions)
        }
        common_metadata={
            "name":"dolma",
            "model":"allenai/OLMo-7B",
            "batch_size":batch_size,
            "start_batch_no":start_batchno,
            "val_batch_no":val_batchno,
            "end_batch_no":end_batchno,
            "train_config_path":train_config_path,
            "file_config_path":data_order_file_path,
        }
        train_metadata={
            **common_metadata,
            "split":"train",
        }
        valid_metadata={
            **common_metadata,
            "split":"train",
        }

        dataset_with_metadata = DatasetDictWithMetadata(
            datasets = {"member":train_data,"nonmember":valid_data},
            metadata = {"member":train_metadata,"nonmember":valid_metadata}
        )        
        if save_to_hf_file:
            dataset_with_metadata.push_to_hub(save_to_hf_file)
        return dataset_with_metadata

