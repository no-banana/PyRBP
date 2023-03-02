.. PyRBP documentation master file, created by
   sphinx-quickstart on Sun Oct 23 20:29:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRBP's documentation!
=========================================================
**Date:** March 2, 2023. **Version:** 0.1.0

**paper:** PyRBP: A Python Framework for Reliable Identification and Characterization of High-Throughput RNA-Binding Protein Events.

**Citing Us:**

If you find PyRBP helpful in your work or research, we would greatly appreciate citations to the following paper
::
 put the bib here

PyRBP is a Python toolbox for quick generation, condensation, evaluation, and visualization of different features for RBP sequence data. It was built on the basis of scikit-learn and tensorflow. PyRBP includes three types of features, from the classical biological properties (seven categories) and semantic information (five categories) to secondary structure features.

**PyRBP is featured for:**

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

.. toctree::
   :maxdepth: 2
   :caption: EXAMPLES

   ./basic_usage.rst
   ./fileOperation_examples.rst
   ./featureGeneration_examples.rst
   ./featureSelection_examples.rst
   ./evaluateClassifiers_examples.rst
   ./plotAnalysis_examples.rst


.. toctree::
   :maxdepth: 2
   :caption: HISTORY

   ./release_history.rst


