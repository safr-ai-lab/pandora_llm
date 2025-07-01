# Initialize classes

from pandora_llm.data import DatasetDictWithMetadata, DatasetWithMetadata, load_dataset_with_metadata
import numpy as np
from datasets import Dataset# , DatasetDict
import torch
from tqdm import tqdm
import os
from transformers import AutoConfig, AutoTokenizer
tokenizer_name = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Before and after
pt_file = "pile_98_text.pt" # User can modify this name
nonmember = "/n/holyscratch01/sneel_lab/jgwang/pile/aria2-1.37.0-linux-gnu-64bit-build1/pythia/intermediate_npys/step98001_98026/indicies.npy"
nmem = np.load(nonmember).tolist()
member = "/n/holyscratch01/sneel_lab/jgwang/pile/aria2-1.37.0-linux-gnu-64bit-build1/pythia/intermediate_npys/step97974_97999/indicies.npy"
mem = np.load(member).tolist()

if os.path.exists(pt_file):
    print(f"Loading cached text from {pt_file}")
    saved_data = torch.load(pt_file)
    memtext = saved_data['memtext']
    nmemtext = saved_data['nmemtext']
else:
    print("Generating text from indices...")
    nmemtext = []
    for row in tqdm(nmem):
           nmemtext.append(tokenizer.decode(row))

    memtext = []
    for row in tqdm(mem):
           memtext.append(tokenizer.decode(row))
    
    # Save the generated text
    print(f"Saving generated text to {pt_file}")
    torch.save({
        'memtext': memtext,
        'nmemtext': nmemtext
    }, pt_file)

memdataset = Dataset.from_dict({'tokens': mem, 'text': memtext})
nonmemdataset = Dataset.from_dict({'tokens': nmem, 'text': nmemtext})

datadict = {'member': memdataset, 'nonmember': nonmemdataset}
metadata_test = {
    'member': {'step': 'step97974_97999', 'additional_info': 'value1'},
    'nonmember': {'step': 'step98001_98026', 'additional_info': 'value2'}
}

# Create object
obj = DatasetDictWithMetadata(datasets=datadict, metadata=metadata_test)
print(obj)
obj.save_to_disk("step98dd")
obj.push_to_hub("jeffreygwang/testdata")


"""
{'member': array([[  521,   259, 21315, ..., 29852,   105, 42335],
       [ 3549,     5,    17, ...,   347,   470,    15],
       [ 4722,   281,  3877, ..., 10502,  2464,   187],
       ...,
       [ 4677,   562,  4555, ..., 16493,   401,    15],
       [   80,  2153,    13, ...,  3866,    31,   187],
       [ 3718,   275,   253, ...,    15,   346,  8486]], dtype=uint16), 
'nonmember': array([[  253,  6420,    14, ...,    13,   326,    13],
       [  688, 15628,   806, ...,   689,   673,    15],
       [  309,  1869,   253, ...,    15,  1500,  6726],
       ...,
       [ 1273,  4248,    13, ..., 16055, 36870,    13],
       [   15,  3015, 34892, ..., 17352,  6798,   411],
       [   41, 20878,   296, ...,  1047,    93,    20]], dtype=uint16)}
"""





