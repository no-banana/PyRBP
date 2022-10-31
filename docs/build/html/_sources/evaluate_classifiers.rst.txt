RBP_package.evaluateClassifiers
=============================================

This two functions are used to evaluate the validity of the sequence representations obtained in the preceding process.

.. py:function:: RBP_package.evaluateClassifiers.evaluateDLclassifers(features, labels, file_path='', shuffle=True, folds=5)

    ``RBP_package`` integrates four classical deep learning models (CNN, RNN, MLP and ResNet), cross-validates them using the representation matrix on the four classes of models, and stores the final performance metrics obtained for each model in ``DL_evalution_metrics.csv``.

.. py:function:: RBP_package.evaluateClassifiers.evaluateMLclassifers(features, labels, file_path='', shuffle=True, folds=5)

    ``RBP_package`` integrates four classical deep learning models (CNN, RNN, MLP and ResNet), cross-validates them using the representation matrix on the four classes of models, and stores the final performance metrics obtained for each model in ``ML_evalution_metrics.csv``.


    