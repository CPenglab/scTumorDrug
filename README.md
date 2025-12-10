# scTumorDrug
In this work, we integrated the labeled scRNA-seq, bulk RNA and drug response data to develop the computational tool scTumorDrug to predict cell-type specific drug responses for heterogeneous tumors. We systematically evaluated scTumorDrug by using different kinds of public datasets, including cell lines, mouse model and clinical human samples.

## Installation
The software is a stand-alone python script package. The home directory of scTumorDrug can be cloned from the GitHub repository:

```
# Clone from Github 
git clone https://github.com/CPenglab/scTumorDrug.git
# Go into the directory 
cd scTumorDrug
```

## example usage:
```
python model/scTumorDrug.py -id AML -d I-BET151 -base CTRP
```  
`-id` refers to the single-cell data input that needs to be predicted, with row names as barcodes and column names as genes. The input format can refer to data/Input_data_processing.txt, which is the code used to extract the counts file from Seurat.  
`-d` refers to the input drug that needs to be predicted. The drug name format can refer to the drug names in data/CTRP2_Res_nAUC.txt and data/GDSC2_Res_nAUC.txt.  
`-base` refers to selecting which database to use for training the model. The options include CTRP and GDSC.

## Output
The output results are saved in results/{`id`}_{`d`}.csv, with the format consisting of one column for barcodes and another column for drug sensitivity labels. For integrating the results into the Seurat object, please refer to data/Output_data_integration.txt.
