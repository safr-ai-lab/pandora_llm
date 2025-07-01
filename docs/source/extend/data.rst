Data
====

Language data comes in two forms: text (which are universal) and input ids (which are tokenizer-specific).

We always prefer to go from text to input ids, since the ultimate goal is to recover the plaintext. Note that most tokenizers are NOT bijective. 

In general, our API accepts HuggingFace datasets with a `member` and `nonmember` split, each having two columns: `text` and `tokens`.

The Pile
--------


Dolma
-------

This library provides tools to generate OLMO training data for membership inference attacks. The process creates datasets by splitting OLMO training data at a specific batch number to distinguish between "member" (pre-split) and "non-member" (post-split) data.

Prerequisites:

Before generating OLMO data, you'll need:

- OLMO training configuration file (`OLMo-7B-local.yaml`)
- Access to OLMO global indices file

Data Download:

First, download the required OLMO preprocessed data files:

.. code-block:: bash
    
    bash scripts/data_generation/wget_olmo_data.sh

This script downloads `.npy` files containing the preprocessed OLMO training data.

Generating Training Data:

Use the data generation script to create member/non-member datasets:

.. code-block:: bash

    python scripts/data_generation/create_olmo_hf_data.py

You can also use the core data creation function directly with custom parameters:

.. code-block:: bash
    python src/pandora_llm/data/create_olmo_data.py \
        --data_order_file_path "https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy" \
        --train_config_path "src/pandora_llm/data/OLMo-7B-local.yaml" \
        --model_name "OLMO7B-local" \
        --batchno 400000 \
        --num_data 10000 \
        --seed 229


Parameters:

Key parameters for OLMO data generation are as follows:

- `--data_order_file_path`: URL or path to the global indices file
- `--train_config_path`: Path to the OLMO training configuration YAML file pointing to `.npy` files 
- `--model_name`: Name identifier for the generated dataset
- `--start_batchno`: Starting batch number for member data (default: 0)
- `--batchno`: Split point batch number (default: 400000)
- `--end_batchno`: Ending batch number for non-member data (default: auto-detected)
- `--num_data`: Number of samples to generate for each split
- `--seed`: Random seed for reproducibility

Output:

The data generation process produces

1. **Local PyTorch files**: Saved in the `Data/` directory with names like:
   - `OLMO7B-local_num_data=10000_start=0_middle=400000_end=432410_bs=2160_train.pt`
   - `OLMO7B-local_num_data=10000_start=0_middle=400000_end=432410_bs=2160_valid.pt`

2. **HuggingFace Dataset**: Automatically pushed to HuggingFace Hub (if configured) 

3. **Processed Format**: Each dataset contains:
   - `tokens`: Token IDs for each sample
   - `text`: Decoded text strings
   - Metadata with generation parameters

Preloaded Datasets:

For convenience, we have already preloaded some [OLMO datasets](https://huggingface.co/datasets/mfli314/OLMo-7B-mia-n10000) that can be used immediately without needing to generate new data:

- **`mfli314/OLMo-7B-mia-n10000`**: Contains 10,000 samples split at batch 400,000, compatible with OLMo-7B model at revision `step400000-tokens1769B`