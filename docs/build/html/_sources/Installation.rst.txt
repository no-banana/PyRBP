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
    pip install -r requirement.txt

Choose the appropriate torch version for your own devices

::

    pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


skfeature installation

::

    # for linux
    git clone https://github.com/jundongl/scikit-feature.git
    cd scikit-feature
    python setup.py install

    # for Windows
    git clone https://github.com/jundongl/scikit-feature.git
    cd scikit-feature
    setup.py install