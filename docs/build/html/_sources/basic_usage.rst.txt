RBP_package basic usage flow
=================================
This example illustrates the basic usage of ``RBP_package``, including loading the dataset, generating features, feature selection, training the model, and performance and feature analysis.

This example uses:

- ``RBP_package.filesOperation``
- ``RBP_package.Features``
- ``RBP_package.evaluateClassifiers``
- ``RBP_package.metricsPlot``
- ``RBP_package.featureSelection``

.. code-block:: py

    from RBP_package.filesOperation import read_fasta_file, read_label
    from RBP_package.Features import generateDynamicLMFeatures, generateStaticLMFeatures, generateStructureFeatures, generateBPFeatures
    from RBP_package.evaluateClassifiers import evaluateDLclassifers
    from RBP_package.metricsPlot import violinplot, shap_interaction_scatter
    from RBP_package.featureSelection import cife
    from sklearn.svm import SVC

Load the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load a AGO1 dataset as example.

.. code-block:: py

    # Define the path where the dataset locates.
    fasta_path = '/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/seq'
    label_path = '/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/label'

    sequences = read_fasta_file(fasta_path)  # Read sequences and labels from given path
    label = read_label(label_path)

Generate features for sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We generate three types of features as examples, in generating biological features, we generate Positional gapped k-m-tuple pairs (PGKM) features, in generating semantic information, we process the sequence as 4mer in dynamic model, while in static model, we process the sequence as 3mer and use fasttext as the model for embedding extraction.

.. code-block:: py

    biological_features = generateBPFeatures(sequences, PGKM=True)  # generate biological features
    bert_features = generateDynamicLMFeatures(sequences, kmer=4, model='/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/dynamicRNALM/circleRNA/pytorch_model_4mer')  # generate dynamic semantic information
    static_features = generateStaticLMFeatures(sequences, kmer=3, model='/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/staticRNALM/circleRNA/circRNA_3mer_fasttext') # static semantic information
    structure_features = generateStructureFeatures(fasta_path, script_path='/home/wangyansong/RBP_package_test/src/RBP_package/RNAplfold', basic_path='/home/wangyansong/RBP_package_test/src/circRNAdatasetAGO1', W=101, L=70, u=1)  # generate secondary structure information

Perform feature selection to refine the biological features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We take the cife method as example.

.. code-block:: py

    refined_biological_features = cife(biological_features, label, num_features=10)  # refine the biologcial_feature using cife feature selection method




