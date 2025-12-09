import pandas as pd
import numpy as np
import os
import torch
import random



# Set the random seed
torch.manual_seed(2002)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2002)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

np.random.seed(2002)
random.seed(2002)
# parameters
import argparse
parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--GSEid','-id',help='Data Name',default='AML')
parser.add_argument('--Drug','-d',help='drug name',default='I-BET151')
parser.add_argument('--Database','-base',help='Choose a cancer cell line database',default='CTRP')
parser.add_argument('--Gene_num','-g',help='Number of genes',default=1500,type=int)
#
parser.add_argument('--h_dim1','-h_dim1',help='hidden_dim1',type=int,default=256)
parser.add_argument('--h_dim2','-h_dim2',help='hidden_dim2',type=int,default=128)
parser.add_argument('--h_dim3','-h_dim3',help='hidden_dim3',type=int,default=32)
parser.add_argument('--o_dim','-output_dim',help='output_dim',type=int,default=2)
parser.add_argument('--dropout','-dropout',help='dropout',type=float,default=0.2)
parser.add_argument('--epochs','-ep',help='epochs',type=int,default=300)
parser.add_argument('--l_r','-l_r',help='Learning rate',type=float,default=0.0005)
parser.add_argument('--a1','-a1',help='Reconstruct the bulk loss',type=float,default=0.1)
parser.add_argument('--a2','-a2',help='Reconstruct the scRNA loss',type=float,default=0.1)
parser.add_argument('--b1','-b1',help='Binary classification loss for Bulk',type=float,default=1)
parser.add_argument('--b2','-b2',help='Binary classification loss for CCL_ScRNA',type=float,default=1)
parser.add_argument('--c','-c',help='Loss for Domain Adaptation',type=float,default=0.01)

#
args = parser.parse_args()
GSEid = args.GSEid
Drug = args.Drug
Database = args.Database
Gene_num = args.Gene_num
hidden_dim1 = args.h_dim1
hidden_dim2 = args.h_dim2
hidden_dim3 = args.h_dim3
output_dim = args.o_dim
dropout = args.dropout
epochs = args.epochs
l_r = args.l_r
a1 = args.a1
a2 = args.a2
b1 = args.b1
b2 = args.b2
c = args.c



print("--- Start running scTumorDrug ---")

os.chdir("/public/home/zhqiang/yard/model/scTumorDrug/model")
from modules import *
from Binary_Label_Classification import *


os.chdir("/public/home/zhqiang/yard/model/scTumorDrug")

dirName = "results"
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

print("--- Load bulk data ---")
if Database == "CTRP":
    Bulk = read_r("data/CCLE_Expr_Log2TPM.rds")[None]
    AUC = read_r("data/CTRP2_Res_nAUC.rds")[None]
    AUC = Binary_Label_Classification(AUC)
elif Database == "GDSC":
    Bulk = read_r("data/GDSC_Expr_CVCL.rds")[None]
    AUC = read_r("data/GDSC2_Res_nAUC.rds")[None]
    AUC = Binary_Label_Classification(AUC)
else:
    raise NotImplementedError('Database {} not match'.format(Database))

print("--- Load CCL_ScRNA ---")
CCL_ScRNA = read_r('data/GSE157220_CPM_matrix.rds')[None]
CCL_ScRNA_meta = pd.read_csv('data/GSE157220_PAN_meta.csv',index_col=0)

CCL_ScRNA = CCL_ScRNA.transpose()
CCL_ScRNA = np.log2(CCL_ScRNA + 1)
CCL_ScRNA = CCL_ScRNA.transpose()

print("--- Successfully loaded the scRNA data of the cancer cell line ---")

print("--- Load scRNA ---")
if GSEid == "Breast":
    SC_RNA = pd.read_csv("data/scdata/Breast-MCF7/GSE114461_Bortezomib_processed.csv.gz",index_col=0, compression='gzip').T
    SC_RNA = log2TPM_1(SC_RNA)
elif GSEid == "AML":
    SC_RNA = pd.read_csv("data/scdata/AML-PDX/GSE110894_expr_count_converted.csv.gz.gz",index_col=0, compression='gzip')
    labels_dict = dict()
    sc_label = pd.read_csv("data/scdata/AML-PDX/GSE110894_SingleCellInfo.csv", header=3).dropna()
    for plate,well,name in zip(sc_label[['Plate#']].values, sc_label[['Well position']].values, sc_label[['Sample name']].values):
        colname = plate[0] + '_' + well[0]
        if 'PARENTALS' in name[0]:
            labels_dict[colname] = 0
        elif 'RESISTANT' in name[0]:
            labels_dict[colname] = 1
        else:
            continue
    sc_idx = np.where(SC_RNA.columns.isin(list(labels_dict.keys())))[0]
    SC_RNA = SC_RNA.iloc[:, sc_idx]
    SC_RNA = log2TPM_1(SC_RNA)
elif GSEid == "GSE192575_human":
    Sen = pd.read_csv("scData/GSE192575_human/Sen_count.csv",index_col=0).T
    Res = pd.read_csv("scData/GSE192575_human/Res_count.csv",index_col=0).T
    Sen = log2TPM_2(Sen)
    Res = log2TPM_2(Res)
    samples = [Sen,  Res]  
    gene_lists = [set(df.columns) for df in samples]
    common_genes = sorted(list(set.intersection(*gene_lists)))
    aligned_samples = []
    for df in samples:
        print(df)
        df_aligned = df.reindex(columns=common_genes, fill_value=0)
        aligned_samples.append(df_aligned)
    combined_data = pd.concat(aligned_samples, axis=0)
    SC_RNA = combined_data.T
else:
    raise NotImplementedError('GSEid {} not match'.format(GSEid))

print("--- Successfully loaded scRNA data","---------------","Start training ---")
shared_encoder, model,filtered_SC_RNA ,sc= train(Bulk,AUC,CCL_ScRNA,CCL_ScRNA_meta,SC_RNA,Drug,Database,l_r,a1,a2,b1,b2,c,Gene_num,epochs,hidden_dim1,hidden_dim2,hidden_dim3,output_dim,dropout)


z_sc = shared_encoder(sc)
sc_NN_outputs = model(z_sc)
sc_NN_outputs = torch.argmax(sc_NN_outputs, dim=1)
sc_NN_outputs_np = sc_NN_outputs.detach().numpy().flatten()
sc_NN_outputs_series = pd.Series(sc_NN_outputs_np, index=filtered_SC_RNA.index)
sc_NN_outputs_series.to_csv(dirName+'/{}_{}.csv'.format(GSEid, Drug), index=True)

print("---  successfully saved "+dirName+"/{}_{}.csv".format(GSEid, Drug))

































