from __future__ import annotations
import os
import textwrap
from typing import Collection, Union, Iterable
import warnings
import numpy as np
import pandas as pd
import torch

class FeatureSet(dict):
    """
    A specialized dictionary for storing and managing features with metadata.

    The FeatureSet class maps feature names to their corresponding data, where each feature is 
    represented as an N x d collection (e.g., numpy arrays, pandas DataFrames, torch tensors). 
    It provides utility methods for validating, merging, concatenating, and loading features, 
    while also supporting associated metadata.

    Attributes:
        metadata (dict): Metadata associated with the features, providing additional context.
    """
    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.pop("metadata",{})
        super().__init__(*args, **kwargs)
        self._validate()

    def __repr__(self):
        feature_repr = "\n".join([f"{k}: {type(v).__name__} with shape {getattr(v, 'shape', (len(v),))}" for k, v in self.items()])
        metadata_repr = "\n".join([f"{k}: {v}" for k, v in self.metadata.items()])
        return f"FeatureSet(features={{\n{textwrap.indent(feature_repr,' '*4)}\n}}, metadata={{\n{textwrap.indent(metadata_repr,' '*4)}\n}})"

    @property
    def num_samples(self) -> int:
        return len(next(iter(self.values()), []))

    def _validate(self) -> None:
        """Ensure all features have the same number of samples."""
        if len(self)>0:
            for key, value in self.items():
                if not isinstance(value, Collection):
                    raise ValueError(f"Feature '{key}' must be a collection (got {type(value)}).")
                if not isinstance(value[0], Collection):
                    raise ValueError(f"Feature '{key}' must be a collection of collections (inner got {type(value[0])}).")
                if len(value) != self.num_samples:
                    raise ValueError(f"Feature '{key}' has inconsistent sample size (expected {self.num_samples}, got {len(value)}).")

    def merge(self, other: FeatureSet) -> FeatureSet:
        """
        Merges another FeatureSet into the current FeatureSet, preferring values from the other FeatureSet.

        Args:
            other: The FeatureSet to merge with.

        Returns:
            FeatureSet: A new FeatureSet containing merged features and metadata.
        
        Raises:
            ValueError: If `other` is not a FeatureSet.
        
        Warnings:
            UserWarning: If conflicting feature values are found, a warning is issued.
        """
        if not isinstance(other, FeatureSet):
            raise ValueError("Can only merge with another FeatureSet.")
        merged = FeatureSet(metadata={**self.metadata, **other.metadata})
        for key in set(self.keys()).union(other.keys()):
            if key in self and key in other and not np.array_equal(self[key], other[key]):
                warnings.warn(f"Conflicting values for key '{key}'. Using the value from the second FeatureSet.", UserWarning)
            merged[key] = other.get(key, self.get(key))
        merged._validate()
        return merged

    def concatenate(self, other: FeatureSet) -> FeatureSet:
        """
        Concatenates another FeatureSet along the sample dimension.

        Args:
            other: The FeatureSet to concatenate with.

        Returns:
            FeatureSet: A new FeatureSet with concatenated features and combined metadata.
        
        Raises:
            ValueError: If `other` is not a FeatureSet or if features do not match between the FeatureSets.
            TypeError: If an unsupported feature type is encountered.
        """
        if not isinstance(other, FeatureSet):
            raise ValueError("Can only concatenate with another FeatureSet.")
        if self.keys() != other.keys():
            raise ValueError(f"FeatureSets must have the same features to concatenate. Missing in self: {other.keys() - self.keys()}. Missing in other: {self.keys() - other.keys()}.")
        concatenated = FeatureSet(metadata={**self.metadata, **other.metadata})
        for key in self.keys():
            if isinstance(self[key], np.ndarray):
                concatenated[key] = np.concatenate([self[key], other[key]], axis=0)
            elif isinstance(self[key], pd.DataFrame):
                concatenated[key] =  pd.concat([self[key], other[key]], axis=0, ignore_index=True)
            elif isinstance(self[key], torch.Tensor):
                concatenated[key] =  torch.cat([self[key], other[key]], dim=0)
            elif isinstance(self[key], list):
                concatenated[key] =  self[key] + other[key]
            elif isinstance(self[key], tuple):
                concatenated[key] =  self[key] + other[key]
            else:
                raise TypeError(f"Unsupported feature type for concatenation: {type(self[key])}")
        concatenated._validate()
        return concatenated

    @classmethod
    def load(cls, filenames: Union[str,Iterable[str]], default_names: Iterable[str]=None, method: str="merge") -> FeatureSet:
        """
        Loads data from .pt files and combines them into one FeatureSet.

        Args:
            filenames: A single filename or a list of filenames.
            default_names: Default names for features if raw tensors are loaded. If not provided, names will be derived from filenames.
            method: Specifies how to combine the data. Options are 'merge' or 'concat'.

        Returns:
            FeatureSet: A new FeatureSet containing the combined data.
        
        Raises:
            ValueError: If filenames are invalid, or if `method` is not 'merge' or 'concat'.
            TypeError: If loaded data type is unsupported.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        elif not isinstance(filenames, list) or not all(isinstance(f, str) for f in filenames):
            raise ValueError("Filenames must be a string or a list of strings.")
        if default_names and len(default_names) != len(filenames):
            raise ValueError("Length of default_names must match length of filenames.")
        if method not in {"merge", "concat"}:
            raise ValueError("Method must be either 'merge' or 'concat'.")

        combined_fs = None
        for i, filename in enumerate(filenames):
            data = torch.load(filename)
            if isinstance(data, torch.Tensor):
                feature_name = default_names[i] if default_names else os.path.splitext(os.path.basename(filename))[0]
                data = cls({feature_name: data})
            elif not isinstance(data, FeatureSet):
                raise TypeError(f"Unsupported data type loaded from '{filename}': {type(data)}")
            if combined_fs is None:
                combined_fs = data
            else:
                if method == "merge":
                    combined_fs = combined_fs.merge(data)
                elif method == "concat":
                    combined_fs = combined_fs.concatenate(data)
        return combined_fs