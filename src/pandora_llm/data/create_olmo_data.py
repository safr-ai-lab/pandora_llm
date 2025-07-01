from accelerate.utils import set_seed
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import os 

from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

"""
This script identifies train and test data before and after a batch no. in a training order:
1) python create_olmo_data.py --data_order_file "https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy" \
    --train_config_path /n/holyscratch01/sneel_lab/mfli/dolmo_test/OLMo/configs/official/OLMo-7B-local.yaml \
    --model_name OLMO7B-local --batchno 400000 --num_data 1000 --seed 229 

"""

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    # Data Collection Arguments
    parser.add_argument('--data_order_file_path', action="store", type=str, required=True, help='Data order path')
    parser.add_argument('--train_config_path', action="store", type=str, required=True, help='Train config path')
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Model Name')
    parser.add_argument('--start_batchno', action="store", type=int, required=False,default=0,help='Start batch number')
    parser.add_argument('--batchno', action="store", type=int, required=True, help='Split batch number')
    parser.add_argument('--end_batchno', action="store", type=int, required=False,default=None,help='End batch number')
    parser.add_argument('--num_data', action="store", type=int, required=True, help='Number of training and validation data')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    return parser.parse_args(raw_args)

def create_olmo_data(args):
    data_order_file_path = cached_path(args.data_order_file_path)
    train_config_path = args.train_config_path
    start_batchno = args.start_batchno
    batchno = args.batchno 
    name =  args.model_name
    num_data = args.num_data

    ## Prepare dataset 
    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    batch_size = cfg.global_train_batch_size
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    

    end_batchno = args.end_batchno if args.end_batchno else (len(global_indices)-1) // batch_size
    assert start_batchno < batchno < end_batchno, "Need start_batchno < batchno < end_batchno"
    assert batch_size*batchno < len(global_indices), "Need batchno * batch_size to be smaller than number of samples in first epoch"

    set_seed(args.seed)
    def get_batch_instances(batch_idx: int, batch_index : int) -> list[int]:
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_indices = global_indices[batch_start:batch_end]
        return dataset[batch_indices[batch_index]]["input_ids"].tolist()

    # Collect training and validation data by sampling from batches w/o replacement    
    train_positions  = np.random.choice(list(range(batch_size*start_batchno,
                                                   batch_size*batchno)),
                                        size=num_data,replace=False)
    valid_positions  = np.random.choice(list(range(batch_size*(batchno+10),
                                                   batch_size*(end_batchno))),
                                        size=num_data,replace=False)

    train_batches, train_batch_index    = train_positions // batch_size, train_positions % batch_size
    valid_batches, valid_batch_index    = valid_positions // batch_size, valid_positions % batch_size

    train_data, valid_data = [], []
    print("Collecting training and validation data") 
    for i in tqdm(range(num_data)):
        train_data.append(get_batch_instances(train_batches[i], train_batch_index[i]))
        valid_data.append(get_batch_instances(valid_batches[i], valid_batch_index[i]))

    train_data, valid_data = torch.tensor(train_data), torch.tensor(valid_data)

    # Save Information
    directory_path = "Data"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")  

    # Save train and valid
    finalsavetrain = f"Data/{name}_num_data={num_data}_start={start_batchno}_middle={batchno}_end={end_batchno}_bs={batch_size}_train.pt"
    finalsaveval = f"Data/{name}_num_data={num_data}_start={start_batchno}_middle={batchno}_end={end_batchno}_bs={batch_size}_valid.pt"

    # Data
    print(f"Saving to... {finalsavetrain} and {finalsaveval}")   
    torch.save({"data" : train_data,
                "name" : name,
                "type" : "train",
                "num_data" : num_data,
                "start_batch_no" : start_batchno,
                "batch_no" : batchno,
                "end_batch_no" : end_batchno,
                "train_config_path" : args.train_config_path,
                "file_config_path"  : args.data_order_file_path,
                "train_batches":train_batches, 
                "train_batch_index":train_batch_index
                }, 
                finalsavetrain
                )
    torch.save({"data" : valid_data,
                "name" : name,
                "type" : "valid",
                "num_data" : num_data,
                "start_batch_no" : start_batchno,
                "batch_no" : batchno,
                "end_batch_no" : end_batchno,
                "train_config_path" : args.train_config_path,
                "file_config_path"  : args.data_order_file_path,
                "valid_batches":valid_batches, 
                "valid_batch_index":valid_batch_index
                },
               finalsaveval
               )

def main(raw_args=None):
    args = get_args(raw_args)
    create_olmo_data(args)

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")
