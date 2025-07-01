Setup Guide
===========

Installation
------------
From pip:

.. code-block:: bash
   
   pip install pandora-llm

From source:

.. code-block:: bash

   git clone https://github.com/safr-ai-lab/pandora_llm.git
   pip install -e .

Our library has been tested on Python 3.10 on Linux with GCC 11.2.0.

Understanding the File Tree
---------------------------

If you installed from source, you will see the following directory structure:

.. literalinclude:: dir_tree.txt

.. note:: Large models tend to fill up disk space quickly. Clean any generated folders periodically, or specify the ``--experiment_name`` and ``--model_cache_dir`` flag with your desired save location.

Building the Docs
-----------------
See ``docs/requirements.txt`` for the required packages.

To make the docs:

.. code-block:: bash

   cd docs
   make html

To live preview the docs:

.. code-block:: bash

   cd docs
   sphinx-autobuild source build/html

Then the docs will be available under ``docs/build/html/index.html``.
