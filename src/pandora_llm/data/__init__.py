from .base.Dataset import DatasetDictWithMetadata, DatasetWithMetadata, load_dataset_with_metadata, concatenate_datasets_with_metadata
from .base.utils import collate_fn, ChainWithLength

# For now, because these files are broken
from .ThePile import ThePile
#from .Dolma import Dolma