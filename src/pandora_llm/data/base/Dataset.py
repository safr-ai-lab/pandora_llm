from __future__ import annotations
import os
import io
import re
import json
from typing import Union
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from datasets.table import Table
from datasets.utils.typing import PathLike
from huggingface_hub import HfApi, hf_hub_download

class DatasetDictWithMetadata(DatasetDict):
    """
    A HuggingFace DatasetDict with metadata support.
    Metadata is stored in the .metadata dict and is saved/pushed to Hub via 'metadata.json'.
    """
    def __init__(self, datasets: dict[str, Union[Dataset,Table]]={}, metadata: dict={}):
        """
        Initialize the DatasetDictWithMetadata.
        Args:
            datasets: A dictionary of dataset splits (e.g., {'train': Dataset, 'test': Dataset}).
        """
        for key,dataset in datasets.items():
            if key in metadata:
                datasets[key] = DatasetWithMetadata(dataset,**metadata[key])
        super().__init__(datasets)

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"DatasetDictWithMetadata({{\n{repr}\n}})"

    @classmethod
    def load_from_disk(cls, dataset_path: PathLike, **kwargs) -> DatasetDictWithMetadata:
        """
        Load the DatasetDict and metadata from disk.
        Args:
            dataset_path: Path to load the dataset.
        Returns:
            DatasetDictWithMetadata: The loaded dataset dictionary with metadata.
        """
        datasets = DatasetDict.load_from_disk(dataset_path, **kwargs)
        metadata = {}
        for split in datasets.keys():
            metadata_path = f"{dataset_path}/{split}/metadata.json"
            try:
                with open(metadata_path, "r") as f:
                    metadata[split] = json.load(f)
            except FileNotFoundError:
                metadata[split] = {}

        return cls(datasets, metadata)

    def push_to_hub(self, repo_id, **kwargs) -> None:
        """
        Push the DatasetDict and metadata to the Hugging Face Hub.
        Args:
            repo_id: The Hugging Face Hub repository ID.
        """
        super().push_to_hub(repo_id, **kwargs)

        for split, dataset in self.items():
            if isinstance(dataset, DatasetWithMetadata):
                metadata_content = json.dumps(dataset.metadata, indent=2)
                api = HfApi()
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=f"{split}/metadata.json",
                    path_or_fileobj=io.BytesIO(metadata_content.encode("utf-8")),
                )
    
    def flatten(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().flatten(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})
    
    def cast(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().cast(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def cast_column(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().cast_column(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def remove_columns(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().remove_columns(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})
    
    def rename_column(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().rename_column(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def rename_columns(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().rename_columns(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})
    
    def select_columns(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().select_columns(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def class_encode_column(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().class_encode_column(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})
    
    def map(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().map(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def filter(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().filter(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})
    
    def flatten_indices(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().flatten_indiceser(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def sort(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().sort(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def shuffle(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().shuffle(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    def align_labels_with_mapping(self, *args, **kwargs):
        return DatasetDictWithMetadata(super().align_labels_with_mapping(*args, **kwargs),metadata={k:self[k].metadata for k in self.keys()})

    @staticmethod
    def from_csv(*args, **kwargs):
        return DatasetDictWithMetadata(super().from_csv(*args, **kwargs))
    
    @staticmethod
    def from_json(*args, **kwargs):
        return DatasetDictWithMetadata(super().from_json(*args, **kwargs))
    
    @staticmethod
    def from_parquet(*args, **kwargs):
        return DatasetDictWithMetadata(super().from_parquet(*args, **kwargs))
    
    @staticmethod
    def from_text(*args, **kwargs):
        return DatasetDictWithMetadata(super().from_text(*args, **kwargs))
    
class DatasetWithMetadata(Dataset):
    """
    A HuggingFace dataset with metadata support.
    Metadata is stored in the .metadata dict and is saved/pushed to Hub via 'metadata.json'.
    """ 
    def __init__(self, data: Union[Dataset,], **kwargs):
        """
        Initialize the dataset with data and metadata.
        Args:
            data: The dataset data (e.g., a Dataset, a dictionary, or Pandas DataFrame).
            kwargs: Optional metadata.
        """
        if isinstance(data, Dataset):
            super().__init__(data._data)
        else:
            super().__init__(data)
        self.metadata = kwargs

    def __repr__(self):
        return f"DatasetWithMetadata({{\n" \
               f"    features: {list(self._info.features.keys())},\n" \
               f"    num_rows: {self.num_rows},\n" + \
               "".join(f"    {key}: {repr(value)},\n" for key,value in self.metadata.items()) + \
               f"}})"

    def save_to_disk(self, dataset_path, **kwargs):
        """
        Save the dataset and metadata to disk.
        Args:
            dataset_path: Path to save the dataset.
        """
        super().save_to_disk(dataset_path, **kwargs)
        metadata_path = f"{dataset_path}/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load_from_disk(cls, dataset_path, **kwargs):
        """
        Load the dataset and metadata from disk.
        Args:
            dataset_path: Path to load the dataset.
        Returns:
            DatasetWithMetadata: The loaded dataset with metadata.
        """
        dataset = Dataset.load_from_disk(dataset_path, **kwargs)

        metadata_path = f"{dataset_path}/metadata.json"
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        return cls(dataset.data, **metadata)

    def push_to_hub(self, repo_id, **kwargs):
        """
        Push the dataset and metadata to Hugging Face Hub.
        Args:
            repo_id: The Hugging Face Hub repository ID.
        """
        super().push_to_hub(repo_id, **kwargs)

        metadata_content = json.dumps(self.metadata, indent=2)
        api = HfApi()
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=f"{kwargs.pop('split',str(self.split) if self.split is not None else 'train')}/metadata.json",
            path_or_fileobj=io.BytesIO(metadata_content.encode("utf-8")),
        )

    def _select_contiguous(self, *args, **kwargs):
        return DatasetWithMetadata(super()._select_contiguous(*args, **kwargs),**self.metadata)   
    
    def _new_dataset_with_indices(self, *args, **kwargs):
        return DatasetWithMetadata(super()._new_dataset_with_indices(*args, **kwargs),**self.metadata)

    def map(self, *args, **kwargs):
        return DatasetWithMetadata(super().map(*args, **kwargs),**self.metadata)

    def train_test_split(self, *args, **kwargs):
        dataset_dict = super().train_test_split(*args, **kwargs)
        return DatasetDictWithMetadata(dataset_dict,metadata={k:self.metadata for k in dataset_dict.keys()})

    def add_column(self, *args, **kwargs):
        return DatasetWithMetadata(super().add_column(*args, **kwargs),**self.metadata)
    
    def add_item(self, *args, **kwargs):
        return DatasetWithMetadata(super().add_item(*args, **kwargs),**self.metadata)
    
    @staticmethod
    def from_csv(*args, **kwargs):
        return DatasetWithMetadata(super().from_csv(*args, **kwargs))

    @staticmethod
    def from_generator(*args, **kwargs):
        return DatasetWithMetadata(super().from_generator(*args, **kwargs))

    @staticmethod
    def from_json(*args, **kwargs):
        return DatasetWithMetadata(super().from_json(*args, **kwargs))

    @staticmethod
    def from_parquet(*args, **kwargs):
        return DatasetWithMetadata(super().from_parquet(*args, **kwargs))

    @staticmethod
    def from_text(*args, **kwargs):
        return DatasetWithMetadata(super().from_text(*args, **kwargs))

    @staticmethod
    def from_spark(*args, **kwargs):
        return DatasetWithMetadata(super().from_spark(*args, **kwargs))

    @staticmethod
    def from_sql(*args, **kwargs):
        return DatasetWithMetadata(super().from_sql(*args, **kwargs))

def load_dataset_with_metadata(path: PathLike, **kwargs) -> DatasetWithMetadata:
    """
    Load a dataset from the Hugging Face Hub or local path and load metadata.json if it exists.

    Args:
        path (str): The dataset repository ID or local path.
        **kwargs: Additional arguments for `datasets.load_dataset`.

    Returns:
        DatasetWithMetadata: A dataset with metadata loaded from metadata.json, if available.
    """
    if os.path.exists(path):
        dataset = load_from_disk(path, **kwargs)
    else:
        dataset = load_dataset(path, **kwargs)
    if isinstance(dataset,Dataset):
        metadata = {}
        if os.path.isdir(path):
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
        else:
            try:
                metadata_path = hf_hub_download(
                    repo_id=path,
                    filename=f"{dataset.split}/metadata.json",
                    repo_type="dataset",
                )
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not find metadata.json in {path} ({str(e)})")
        return DatasetWithMetadata(dataset, **metadata)
    else:
        metadata = {}
        for split in dataset.keys():
            metadata[split] = {}
            if os.path.isdir(path):
                metadata_path = os.path.join(path, split, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata[split] = json.load(f)
            else:
                try:
                    metadata_path = hf_hub_download(
                        repo_id=path,
                        filename=f"{split}/metadata.json",
                        repo_type="dataset",
                    )
                    with open(metadata_path, "r") as f:
                        metadata[split] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not find metadata.json in {path} ({str(e)})")
        return DatasetDictWithMetadata(dataset, metadata)

def concatenate_datasets_with_metadata(*args, **kwargs) -> DatasetWithMetadata:
    return DatasetWithMetadata(concatenate_datasets(*args, **kwargs))