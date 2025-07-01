"""
Spec:
- For every step X that is a multiple of 500 from 0 to 98500, we sampled data from X-25 to X-10, and then X+10 to X+25. 
For 0, we didn't sample before; for 98500, we didn't sample after. 

- For every S in {50k, 70k, 90k, 97k}, we extract member/non-member data in two ways:
    - "Chunk": we sample 15k points randomly from S +/- 25 to S +/- 10
    - "Uniform": we look at all data from 0 to S (or S to 98500) and do reservoir sampling to get a uniformly random (non)member sample.
    - We then create member {chunk, uniform} x non-member {chunk, uniform} datasets for each S.

There are 4 datasets per S, and each one is ~1GB, so this is ~16GB of data. 
"""

import random
from pandora_llm.datasets import DatasetDictWithMetadata, DatasetWithMetadata, load_dataset_with_metadata
import numpy as np
from datasets import Dataset# , DatasetDict
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
import os

ENDSTEP = 98500
NSELECT = 15000
X = [97000, 90000, 70000, 50000]
PATH = "/n/netscratch/sneel_lab/Lab/jgwang/pythiapile/aria2-1.37.0-linux-gnu-64bit-build1/pythia/intermediate_npys_final"


def randSelectSingle(nparray): 
    """
    Given np array, select num_elems random elements w/o replacement
    """
    selected_indices = np.random.choice(nparray.shape[0], NSELECT, replace=False)
    sampled_array = nparray[selected_indices, :]
    print(f"Sample Array Shape: {sampled_array.shape}")
    return sampled_array

def reservoir_sample(step: int, steps: list[str], files: list[str]):
    """
    files: list of file paths (step_X/indices.npy) for X < Y
    seed:  optional random seed for reproducibility

    Returns a list of lists (all ints). 
    """
    reservoir = []
    total_seen = 0  # how many elements we've processed so far
    source_steps = [0 for i in range(NSELECT)] # step that each sample came from

    for i, fpath in enumerate(tqdm(files, desc=f"Reservoir sampling for step {step}.")):
        curr_step = steps[i]
        arr = np.load(fpath) # shape: (15361, 2049)
        for row in arr: # reservoir sampling 
            total_seen += 1
            if len(reservoir) < NSELECT: # still room in the reservoir
                reservoir.append(row)
                # source_steps.append(curr_step) # bug
            else: # reservoir is full, decide whether to replace
                r = random.randint(1, total_seen)
                if r <= NSELECT:
                    reservoir[r-1] = row  # replace the r-th item in the reservoir
                    source_steps[r] = curr_step
        del arr 

    return np.array(reservoir), source_steps, total_seen

def randSelectReservoir(step: int, basepath: str, before: bool = True, seed: int = 229):
    """
    Given a step, perform reservoir sampling of all extracted data before/after that step. 
    """

    def get_files_before(step: int): # get 
        step_dirs = []
        steps = []
        for entry in os.listdir(basepath):
            full_path = os.path.join(basepath, entry)
            if os.path.isdir(full_path) and entry.startswith("step_"):
                step_str = entry.split("_")[1]
                if step_str.isdigit():
                    step_val = int(step_str)
                    if step_val < step:
                        steps.append(step_val)
                        step_dirs.append(os.path.join(os.path.join(basepath, entry), 'indicies.npy'))

        return sorted(step_dirs), steps
    
    def get_files_after(step: int):
        step_dirs = []
        steps = []
        for entry in os.listdir(basepath):
            full_path = os.path.join(basepath, entry)
            if os.path.isdir(full_path) and entry.startswith("step_"):
                step_str = entry.split("_")[1]
                if step_str.isdigit():
                    step_val = int(step_str)
                    if step_val > step and step_val < ENDSTEP:
                        steps.append(step_val)
                        step_dirs.append(os.path.join(os.path.join(basepath, entry), 'indicies.npy'))

        return sorted(step_dirs), steps

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if before:
        files, steps = get_files_before(step)
        return reservoir_sample(step, steps, files)
    else:
        files, steps = get_files_after(step)
        return reservoir_sample(step, steps, files)

def npArrayToDataset(nparray, tokenizer):
    """
    Token processing and text conversion for nparr of token.
    """
    # Tokens
    tokens = nparray.tolist()

    # Text
    strings_list = []
    for idx, token_ids in enumerate(tqdm(nparray, desc="Converting tokens to text")):
        # Convert token IDs to a list of integers
        token_ids_list = token_ids.astype(int).tolist()
        # Decode the token IDs to a string
        decoded_str = tokenizer.decode(token_ids_list)
        strings_list.append(decoded_str)
    
    dataset = Dataset.from_dict({'tokens': tokens, 'text': strings_list})
    return dataset

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    for step in X:
        print(f"Starting data processing for step {step}!")

        # X-25 to X-10
        preA = step - 25
        preB = step - 10
        prepath = os.path.join(PATH, f"step_{preA}_{preB}")
        prenparray = np.load(os.path.join(prepath, "indicies.npy"))[:, :2048]
        print(prenparray.shape)
        prenparray = randSelectSingle(prenparray)
        preXdataset = npArrayToDataset(prenparray, tokenizer)

        # X+10 to X+25
        postA = step + 10
        postB = step + 25
        postpath = os.path.join(PATH, f"step_{postA}_{postB}")
        postnparray = np.load(os.path.join(postpath, "indicies.npy"))[:, :2048]
        print(postnparray.shape)
        postnparray = randSelectSingle(postnparray)
        postXdataset = npArrayToDataset(postnparray, tokenizer)

        print("Beginning reservoir sampling!")

        # 0 to X, reservoir'ed
        preUnifSample, source_steps_pre, total_seen_pre = randSelectReservoir(step, PATH, before=True)
        preXunif = npArrayToDataset(preUnifSample, tokenizer)

        # X to ENDSTEP
        postUnifSample, source_steps_post, total_seen_post = randSelectReservoir(step, PATH, before=False)
        postXunif = npArrayToDataset(postUnifSample, tokenizer)
        # break 

        # Do the four combos of data arrangement
        chunk_chunk = {'member': preXdataset, 'nonmember': postXdataset}
        chunk_chunk_metadata = {
            'member': {'step': step, 'source_steps': None, 'total_seen': None, 'source': f"{preA}_{preB}"},
            'nonmember': {'step': step, 'source_steps': None, 'total_seen': None, 'source': f"{postA}_{postB}"}
        }
        obj = DatasetDictWithMetadata(datasets=chunk_chunk, metadata=chunk_chunk_metadata)
        obj.push_to_hub(f"jeffreygwang/pythia_dedupe_mia_{preA}-{preB}_{postA}-{postB}")
        del obj

        chunk_unif = {'member': preXdataset, 'nonmember': postXunif}
        chunk_unif_metadata = {
            'member': {'step': step, 'source_steps': None, 'total_seen': None, 'source': f"{preA}_{preB}"},
            'nonmember': {'step': step, 'source_steps': source_steps_post, 'total_seen': total_seen_post, 'source': None}
        }
        obj = DatasetDictWithMetadata(datasets=chunk_unif, metadata=chunk_unif_metadata)
        obj.push_to_hub(f"jeffreygwang/pythia_dedupe_mia_{preA}-{preB}_{step}-{ENDSTEP}")
        del obj

        unif_unif = {'member': preXunif, 'nonmember': postXunif}
        unif_unif_metadata = {
            'member': {'step': step, 'source_steps': source_steps_pre, 'total_seen': total_seen_pre, 'source': None},
            'nonmember': {'step': step, 'source_steps': source_steps_post, 'total_seen': total_seen_post, 'source': None}
        }
        obj = DatasetDictWithMetadata(datasets=unif_unif, metadata=unif_unif_metadata)
        obj.push_to_hub(f"jeffreygwang/pythia_dedupe_mia_0-{step}_{step}-{ENDSTEP}")
        del obj

        unif_chunk = {'member': preXunif, 'nonmember': postXdataset}
        unif_chunk_metadata = {
            'member': {'step': step, 'source_steps': source_steps_pre, 'total_seen': total_seen_pre, 'source': None},
            'nonmember': {'step': step, 'source_steps': None, 'total_seen': None, 'source': f"{postA}_{postB}"}
        }
        obj = DatasetDictWithMetadata(datasets=unif_chunk, metadata=unif_chunk_metadata)
        obj.push_to_hub(f"jeffreygwang/pythia_dedupe_mia_0-{step}_{postA}-{postB}")
        del obj

if __name__ == '__main__':
    main()