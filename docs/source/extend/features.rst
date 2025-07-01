Features
========

Features are statistics computed from the data, which often take advantage of the text itself and some varying level of access to a language model.

The base :code:`FeatureComputer` is an abstract class which simply denotes the availability of the :code:`compute_features` method.

Features that involve the language model (which they should) may additionally inherit the :code:`LLMHandler` mixin class, providing :code:`load_model` and :code:`unload_model` functions.

Let us walkthrough ``LOSS`` as an example.

.. code-block:: python

    class LOSS(FeatureComputer, LLMHandler):
        """
        Computes the negative log-likelihood (NLL) for a given dataset using a pre-trained language model.
        Under strong assumptions, thresholding this is approximately optimal by the Neyman-Pearson lemma:
        MALT from Sablayrolles et al. 2019 (https://arxiv.org/pdf/1908.11229).

        Attributes:
            model (AutoModelForCausalLM): The pre-trained language model to compute the NLL.
        """
        def __init__(self, *args, **kwargs):
            FeatureComputer.__init__(self)
            LLMHandler.__init__(self, *args, **kwargs)

        def compute_features(self, dataloader: DataLoader, accelerator: Accelerator) -> Float[torch.Tensor, "n"]:
            """
            Computes the negative log-likelihood (NLL) feature for the given dataloader.

            Args:
                dataloader: The dataloader providing input sequences.
                accelerator: The `Accelerator` object for distributed or mixed-precision training.

            Returns:
                The NLL feature for each sequence in the dataloader.
            
            Raises:
                Exception: If the model is not loaded before calling this method.
            """
            if self.model is None:
                raise Exception("Please call .load_model() to load the model first.")
            return -compute_log_probs_dl(model=self.model,dataloader=dataloader,accelerator=accelerator,mode="mean")

        ...
We observe that the ``LOSS`` class inherits :code:`FeatureComputer` and :code:`LLMHandler`, implementing the :code:`compute_features` abstract method.

The details of the feature computation are hidden within :code:`compute_log_probs_dl` whose details are elided here, but can be easily switched out for any method that accepts the dataloader of input and the language model, outputting features per sample.

We simply recommend that you use the same function signature of computing the attack statistic for a single dataloader for a given number of batches. [#]_

Outside the class (but in the same file), you can define ``compute_log_probs_dl`` or whatever helper functions are necessary to compute your attack statistic.

You can import basic utilities such as computing loss from the ``LOSS`` class or any other attack class, but generally we encourage the attacks to be as self-contained as possible.

.. note::
    Our library assumes that a lower statistic indicates greater confidence to be train data, hence the negative sign.

To use the feature class, simply create an instance of the class with the model name, load the model into memory, compute the statistic, and unload the model when done. It's that easy!

.. code:: python

    # Initialize attack
    LOSSer = LOSS(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Load the model into memory
    LOSSer.load_model()

    # Compute the statistic
    train_statistics = LOSSer.compute_statistic(training_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    val_statistics = LOSSer.compute_statistic(validation_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    
    # Unload when done
    LOSSer.unload_model()

.. rubric:: Footnotes

.. [#] Working with large language models, whether it be inference or training, requires a large amount of computational resources. In this example, we support passing in an ``accelerator`` object from Huggingface `Accelerate <https://huggingface.co/docs/accelerate/index>`_ to automatically handle multi-gpu distributed setups.