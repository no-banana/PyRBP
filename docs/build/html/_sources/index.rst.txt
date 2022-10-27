.. RBP_package_test documentation master file, created by
   sphinx-quickstart on Sun Oct 23 20:29:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RBP_package's documentation!
=========================================================
**Date:** October 25, 2022. **Version:** 0.1.0

**paper:** it depends

**Citing Us:**

If you find RBP_package helpful in your work or research, we would greatly appreciate citations to the following paper
::
 put the bib here

RBP_package is a Python toolbox for quick generation, condensation, evaluation, and visualization of different features for RBP sequence data. It was built on the basis of scikit-learn and tensorflow. RBP_package includes three types of features, from the classical biological properties (seven categories) and semantic information (five categories) to secondary structure features.

**RBP_package is featured for:**

- Unified, easy-to-use APIs, detailed documentation and examples.
- Capable for out-of-the-box one-stop sequencing analysis (feature generation, condensation, model training, performance evaluation, visualization).
- Full compatibility with other popular packages like scikit-learn and yellowbrick.

**API Demo**

.. literalinclude:: ./API_Demo.py

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   ./getting_started.rst
   ./Installation.rst

.. toctree::
   :maxdepth: 2
   :caption: API

   ./file_operation.rst
   ./RNA_features.rst
   ./feature_selection_methods.rst
   ./evaluate_classifiers.rst
   ./analysis_plots.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
