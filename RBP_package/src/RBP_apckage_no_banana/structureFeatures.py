import argparse
import os

def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def defineExperimentPaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    mk_dir(E_path)
    mk_dir(H_path)
    mk_dir(I_path)
    mk_dir(M_path)
    return path, E_path, H_path, I_path, M_path


def definecombinePaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    return path, E_path, H_path, I_path, M_path


def run_RNA(fasta_path, E_path, H_path, I_path, M_path, W, L, u):
    os.system(
        './RNAplfold/E_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        E_path + 'E_profile.txt')
    os.system(
        './RNAplfold/H_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        H_path + 'H_profile.txt')
    os.system(
        './RNAplfold/I_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        I_path + 'I_profile.txt')
    os.system(
        './RNAplfold/M_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        M_path + 'M_profile.txt')

def generateStructureFeatures(dataset_path, basic_path, W, L, u, dataset_name=''):
    path, E_path, H_path, I_path, M_path = defineExperimentPaths(
        basic_path, dataset_name)
    run_RNA(dataset_path, E_path, H_path, I_path, M_path, W=W, L=L, u=u)
    cmd = 'python combine_letter_profiles.py' + ' ' + E_path + 'E_profile.txt' + ' ' + H_path + 'H_profile.txt' + ' ' +\
          I_path + 'I_profile.txt' + ' ' + M_path + 'M_profile.txt' + ' ' + '1' + ' ' + path + 'combined_profile.txt'
    os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--runRNAplfold", action="store_true", help="runRNAplfold")
    args = parser.parse_args()
    args.runRNAplfold = True

    fasta_path = '/home/wangyansong/RBP_package/RBP_package/RNA_datasets/circRNAdataset/AGO1/negative'
    generateStructureFeatures(fasta_path, basic_path='/home/wangyansong/RBP_package/RBP_package/RNA_datasets/circRNAdatasetAGO1', W=101, L=70, u=1)
    # path, E_path, H_path, I_path, M_path = defineExperimentPaths('/home/wangyansong/RBP_package/RBP_package/RNA_datasets/circRNAdataset', 'AGO1')
    # run_RNA(fasta_path, E_path, H_path, I_path, M_path)
    # cmd = 'python combine_letter_profiles.py' + ' ' + E_path + 'E_profile.txt' + ' ' + H_path + 'H_profile.txt' + ' ' + I_path + 'I_profile.txt' + ' ' + M_path + 'M_profile.txt' + ' ' + '1' + ' ' + path + 'combined_profile.txt'
    # os.system(cmd)
