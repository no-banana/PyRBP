from PyRBP.filesOperation import read_fasta_file, read_label
from PyRBP.Features import generateDynamicLMFeatures, generateStaticLMFeatures, generateStructureFeatures, generateBPFeatures
from PyRBP.evaluateClassifiers import evaluateDLclassifers
from PyRBP.metricsPlot import violinplot, shap_interaction_scatter
from PyRBP.featureSelection import cife
from sklearn.svm import SVC

fasta_path = '/home/wangyansong/PyRBP/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/seq'
label_path = '/home/wangyansong/PyRBP/src/RBP_apckage_no_banana/RNA_datasets/circRNAdataset/AGO1/label'

sequences = read_fasta_file(fasta_path)  # read sequences and labels from given path
label = read_label(label_path)

biological_features = generateBPFeatures(sequences, PGKM=True)  # generate biological features
bert_features = generateDynamicLMFeatures(sequences, kmer=4, model='/home/wangyansong/PyRBP/src/dynamicRNALM/circleRNA/pytorch_model_4mer')  # generate dynamic semantic information
static_features = generateStaticLMFeatures(sequences, kmer=3, model='/home/wangyansong/PyRBP/src/staticRNALM/circleRNA/circRNA_3mer_fasttext')
structure_features = generateStructureFeatures(fasta_path, script_path='/home/wangyansong/PyRBP/src/PyRBP/RNAplfold', basic_path='/home/wangyansong/PyRBP/src/circRNAdatasetAGO1', W=101, L=70, u=1)  # generate secondary structure information


refined_biological_features = cife(biological_features, label, num_features=10)  # refine the biologcial_feature using cife feature selection method


evaluateDLclassifers(bert_features, folds=10, labels=label, file_path='./', shuffle=True)  # evaluate CNN, RNN, ResNet-1D and MLP using dynamic semantic information

clf = SVC(probability=True)
shap_interaction_scatter(refined_biological_features, label, clf=clf, sample_size=(0, 100), feature_size=(0, 10), image_path='./')  # Plotting the interaction between biological features in SVM
