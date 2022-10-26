from RBP_package.filesOperation import read_fasta_file, read_label
from RBP_package.biologicalFeatures import generateBPFeatures
from RBP_package.languageFeatures import generateDynamicLMFeatures
from RBP_package.evaluateClassifiers import evaluateDLclassifers
from RBP_package.structureFeatures import generateStructureFeatures
from RBP_package.metricsPlot import violinplot, shap_interaction_scatter
from RBP_package.featureSelection import cife
from sklearn.svm import SVC

fasta_path = '/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/seq'
label_path = '/home/wangyansong/RBP_package/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/label'

sequences = read_fasta_file(fasta_path)  # read sequences and labels from given path
label = read_label(label_path)

biological_features = generateBPFeatures(sequences, PGKM=True)  # generate biological features
bert_features = generateDynamicLMFeatures(sequences, data_type='circRNA', kmer=3)  # generate dynamic semantic information
structure_features = generateStructureFeatures(fasta_path, basic_path='/home/wangyansong/RBP_package_test/src/circRNAdatasetAGO1', W=101, L=70, u=1)  # generate secondary structure information

refined_biological_features = cife(biological_features, label, num_features=10)  # refine the biologcial_feature using cife feature selection method

evaluateDLclassifers(bert_features, folds=10, labels=label, file_path='./', shuffle=True)  # evaluate CNN, RNN, ResNet-1D and MLP using dynamic semantic information

clf = SVC(probability=True)
shap_interaction_scatter(refined_biological_features, label, clf=clf, sample_size=(0, 100), feature_size=(0, 10), image_path='./')  # Plotting the interaction between biological features in SVM