from transformers import AutoModelForCausalLM

class LLMHandler:
    """
    Base class for handling the loading/unloading of large language models. 

    Attributes:
        model (AutoModelForCausalLM): the model to be attacked
        model_name (str): path to the model to be attacked
        model_revision (str): revision of the model to be attacked
        cache_dir (str): directory to cache the model
    """
    def __init__(self, model_name: str, model_revision: str=None, model_cache_dir: str=None):
        """
        Initialize with an attack for a particular model. 

        Args:
            model_name: path to the model to be attacked
            model_revision: revision of the model to be attacked
            cache_dir: directory to cache the model
        """
        self.model = None
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_cache_dir = model_cache_dir
    
    def load_model(self) -> None:
        """
        Loads model into memory
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self) -> None:
        """
        Unloads model from memory
        """
        self.model = None