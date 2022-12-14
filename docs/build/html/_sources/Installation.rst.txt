Installation
=======================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following packages are requirements:

- ``gensim``
- ``glove``
- ``glove_python_binary``
- ``Keras``
- ``matplotlib``
- ``numpy``
- ``scikit_learn``
- ``seaborn``
- ``shap``
- ``tensorflow`` or ``tensorflow_gpu``
- ``torch``
- ``transformers``
- ``yellowbrick``

Install PyRBP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n PyRBP python=3.7.6
    conda activate PyRBP
    git clone https://github.com/no-banana/PyRBP.git
    cd PyRBP
    pip install -r requirements.txt