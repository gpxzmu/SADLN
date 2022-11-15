# SADLN

## Introduction

 SADLN: a self-attention based deep learning network integrating multi-omics data for cancer subtype recognition. SADLN is a middle integration method by consolidates the adversarial generation network and the self-attention mechanism to describe the different distributions of multi-omics data and fusion samples’ relationship. It used an independent sub network to learn omics-specific features and concatenated omics-specific features to an integration representation. Then used a self-attention to learn the relationship of samples on the integration representation and obtained a feature representation that fused the sample relationship. Finally, it used the Gaussian Mixture Model (GMM) to obtain the subtyping label of each sample.

## Files

fea, input: Input multi omics data of SADLN

results: Output of SADLN

==SADLN.py==: Examples of SADLN for subtyping and cluster

~~~python
# the input list of BRCA omics data set is input.txt. We can use the following command to finish the subtyping process: 
python SADLN.py -m SADLN -t BRCA -i ./input/BRCA.list
#the output file are stored in ./results/BRAC.SADLN
~~~

transformer.py: Section of self-attention network 

decisiontree.py: Section of calculating the contribution of single-omics

p_value.R: Section of P-value calculation  in survival analysis

clinlc_analysis.R: Section of clinical parameters enrichment analysis

## Requirements

~~~python
# It is recommended to use the conda command to configure the environment:
conda env create -f environment.yml
~~~

SADLN is based on the Python program language. The generative adversarial network's implementation was based on the open-source library scikit-learn 0.22, Keras 2.2.4, and Tensorflow 1.14.0 .

