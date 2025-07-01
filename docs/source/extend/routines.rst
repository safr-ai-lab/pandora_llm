Routines
========

Now, we can finally put everything together. You can run a membership inference attack through the shortcut :code:`pandora-mia`, which is an alias for :code:`python -m pandora_llm.routines.membership_inference`.

This API takes several important command-line arguments:

- :code:`--dataset.name`: the HuggingFace dataset
- :code:`--dataset.num_train_samples`: number of train samples
- :code:`--dataset.num_val_samples`: number of validation samples
- :code:`--dataset.train_start_index`: index to start taking train samples
- :code:`--dataset.val_start_index`: index to start taking val samples
- :code:`--features.names`: features to use
- :code:`--classifier.name`: classifier to use
- :code:`--model.name`: the HuggingFace language model name
- :code:`--features.compute`: whether to compute the features
- :code:`--classifier.train`: whether to train a classifier
- :code:`--classifier.infer`: whether to run inference

Some attacks will require additional arguments, please refer to the documentation for each attack for more details.

We maintain a collection of minimal example API invocations in our codebase under ``scripts/minimal.sh``.

.. code-block:: bash

   bash scripts/examples.sh