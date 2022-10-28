RBP_package.filesOperation
===================================================

.. py:function:: RBP_package.filesOperation.read_fasta_file(fasta_file='')

    This function is used to read a sequence file in `fasta` format in the following form.
    ::

        >hsa_circ_0000005 start:16578,end:16712
        AGGCGTGGCTACTGCGGCTGGAGCTGCGATGAGACTCGGAACTCCTCGTCTTACTTTGTGCTCCATGTTTTGTTTTTGTATTTTGGTTTGTAAATTTGTAG
        >hsa_circ_0000023 start:50,end:174
        ACCCTTTCTGCCAGCCAGCTAGCCAGGGCCCAGAAACAAACACCGATGGCTTCTTCCCCACGTCCCAAGATGGATGCAATCTTAACTGAGGCCATTAAGGC
        >hsa_circ_0000038 start:189,end:318
        CCGTCCCCCCCACTGCCTACTCATATACCTCCAGAGCCTCCACGCACCCCTCCATTCCCTGCTAAGACTTTTCAAGTTGTGCCAGAAATTGAGTTTCCACC

    :Date: 2001-08-16
    :Version: 1
    :Authors: - Me
              - Myself
              - I
    :Indentation: Since the field marker may be quite long, the second
         and subsequent lines of the field body do not have to line up
         with the first line, but they must be indented relative to the
         field name marker, and they must line up with each other.
    :Parameter i: integer


RBP_package.filesOperation.read_label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test for 2