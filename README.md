<a href="https://pandora-llm.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/website?url=https%3A%2F%2Fpandora-llm.readthedocs.io%2Fen%2Flatest%2F&up_message=sphinx&label=docs&color=blue"></a>
<a href="https://pypi.org/project/pandora-llm/"><img alt="Python version" src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsafr-ai-lab%2Fpandora_llm%2Fblob%2Fmain%2Fpyproject.toml&color=green"></a>
<a href="https://github.com/safr-ai-lab/pandora_llm/blob/main/LICENSE.txt"><img alt="Code license" src="https://img.shields.io/github/license/safr-ai-lab/pandora_llm?color=blue"></a>
<a href="https://github.com/safr-ai-lab/pandora_llm/releases"><img alt="GitHub release" src="https://img.shields.io/github/v/release/safr-ai-lab/pandora_llm?color=green"></a>

<p align="center">
   <img src="docs/source/assets/pandora_llm_title.png" alt="drawing" width="600"/>
</p>

## Overview

`pandora_llm` is a red-teaming library against Large Language Models (LLMs) that assesses their vulnerability to train data leakage.

It provides a unified [PyTorch](https://pytorch.org/) API for evaluating **membership inference attacks (MIAs)**.

Please refer to the [documentation](https://pandora-llm.readthedocs.io/en/latest/) for the API reference as well as tutorials on how to use this codebase.

`pandora_llm` abides by the following core principles:

- **Open Access** — Ensuring that these tools are open-source for all.
- **Reproducible** — Committing to providing all necessary code details to ensure replicability.
- **Self-Contained** — Designing attacks that are self-contained, making it transparent to understand the workings of the method without having to peer through the entire codebase or unnecessary levels of abstraction, and making it easy to contribute new code.
- **Model-Agnostic** — Supporting any [HuggingFace](https://huggingface.co/) model and dataset, making it easy to apply to any situation.
- **Usability** — Prioritizing easy-to-use starter scripts and comprehensive documentation so anyone can effectively use `pandora_llm` regardless of prior background.

We hope that our package serves to guide LLM providers to safety-check their models before release, and to empower the public to hold them accountable to their use of data.

## Installation

From pip:
```
pip install pandora-llm
```

From source:

```bash
git clone https://github.com/safr-ai-lab/pandora-llm.git
pip install -e .
```

## Quickstart
We maintain a collection of starter scripts in our codebase under ``experiments/``. If you are creating a new attack, we recommend making a copy of a starter script for a solid template.

```
python experiments/mia/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229
```

```
bash scripts/run_mia_baselines_olmo.sh
bash scripts/run_mia_baselines_pile.sh
```

## Contributing
We welcome contributions! Please submit pull requests in our [GitHub](https://github.com/safr-ai-lab/pandora-llm).


## Authors

This library was created by Jeffrey G. Wang, Jason Wang, Marvin Li, and Seth Neel.
