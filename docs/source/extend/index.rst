.. _Building with Pandora:

Building with Pandora
=====================

``pandora_llm`` provides a unified open framework for sharing and testing privacy attacks on LLMs.

Our philosophy is to make extending and contributing to ``pandora_llm`` easy by minimizing the effort needed to test a new attack.

Our library decomposes membership inference into the following steps:

.. code-block:: bash
   
   1. Data → 2. Features → 3. Classifier → 4. Metrics


In this section, you will learn how each of these four modules work, and how they combine in the overall routine.

.. toctree::
   :maxdepth: 1

   data
   features
   classifiers
   metrics
   models
   routines