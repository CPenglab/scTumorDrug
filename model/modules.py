import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from pyreadr import read_r
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)


class DomainDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, domain_type='bulk'):
        super().__init__()
        self.domain_type = domain_type
        self.base_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        if domain_type == 'sc':
            self.mean_head = nn.Linear(1024, output_dim)
            self.disp_head = nn.Linear(1024, output_dim)
            self.pi_head = nn.Linear(1024, output_dim)
            self.mean_act = nn.Softplus()
            self.disp_act = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
            self.pi_act = nn.Sigmoid()
        else:
            self.output = nn.Linear(1024, output_dim)
            self.activation = nn.Identity()
    def forward(self, z):
        x = self.base_decoder(z)
        if self.domain_type == 'sc':
            return (
                self.mean_act(self.mean_head(x)),
                self.disp_act(self.disp_head(x)),
                self.pi_act(self.pi_head(x))
            )
        else:
            return self.activation(self.output(x))


class ZINBLoss(nn.Module):
    def __init__(self, scale_factor=1e4):
        super().__init__()
        self.eps = 1e-6
        self.scale_factor = scale_factor
    def forward(self, x, mean, disp, pi):
        scale_factor = self.scale_factor
        mean = mean * scale_factor
        # 
        t1 = torch.lgamma(disp + self.eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + self.eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + self.eps))) + x * (torch.log(disp + self.eps) - torch.log(mean + self.eps))
        nb_loss = t1 + t2
        # 
        zero_nb = torch.pow(disp/(disp + mean + self.eps), disp)
        zero_case = -torch.log(pi + ((1. - pi) * zero_nb) + self.eps)
        nb_case = -torch.log(1. - pi + self.eps) + nb_loss
        result = torch.where(x < 1e-8, zero_case, nb_case)
        return torch.mean(result)




class AdversarialClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.classifier(z)




class MLP(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim2,hidden_dim3, output_dim, dropout):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim1, input_dim2)
        self.fc1 = nn.Linear(input_dim2, hidden_dim2)
        self.bn1 = nn.BatchNorm1d(hidden_dim2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn2 = nn.BatchNorm1d(hidden_dim3)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.dropout1(relu(bn1(fc1(x))))
        x = self.dropout2(relu(bn2(fc2(x))))
        x = self.sigmoid(fc3(x))
        return x



def train(Bulk,AUC,CCL_ScRNA,CCL_ScRNA_meta,SC_RNA,Drug,Database,l_r,a1,a2,b1,b2,c,Gene_num,epochs,hidden_dim1,hidden_dim2,hidden_dim3,output_dim,dropout):
    if Database == "GDSC":
        Drug_genes = pd.read_table("data/Drug_Gene/gdsc/"+"g2_"+Drug+'_signature.txt', sep='\s+')
    elif Database == "CTRP":
        Drug_genes = pd.read_table("data/Drug_Gene/ctrp/"+Drug+'_signature.txt', sep='\s+')
    Drug_gene = Drug_genes.loc[:,"x"].head(Gene_num)
    common_genes_Bulk = Bulk.index.intersection(Drug_gene)
    common_genes_CCL_ScRNA = CCL_ScRNA.index.intersection(Drug_gene)
    common_genes_SC_RNA = SC_RNA.index.intersection(Drug_gene)
    final_common_genes = common_genes_Bulk.intersection(common_genes_CCL_ScRNA)
    final_common_genes = final_common_genes.intersection(common_genes_SC_RNA)
    filtered_Bulk = Bulk.loc[final_common_genes]
    filtered_SC_RNA = SC_RNA.loc[final_common_genes].transpose()
    row_sums = filtered_SC_RNA.sum(axis=1)
    zero_sum_rows = row_sums[row_sums == 0].index
    filtered_SC_RNA = filtered_SC_RNA.loc[row_sums != 0]
    filtered_CCL_ScRNA = CCL_ScRNA.loc[final_common_genes].transpose()
    filtered_SC_RNA.shape[1]
    filtered_CCL_ScRNA.shape[1]
    filtered_Bulk.shape[0]
    Drug_AUC = AUC.loc[:,Drug]
    Drug_AUC_cleaned = Drug_AUC.dropna().to_frame(name='AUC')
    common_columns = filtered_Bulk.columns.intersection(Drug_AUC_cleaned.index)
    filtered_Bulk = filtered_Bulk[common_columns]
    Drug_AUC_cleaned1 = Drug_AUC_cleaned.loc[common_columns,:]
    label = Drug_AUC_cleaned1['AUC'].values
    filtered_Bulk_T = pd.DataFrame(filtered_Bulk.T) 
    filtered_Bulk_T['label'] = label
    Bulk_X = filtered_Bulk_T.drop(columns=['label'])  
    Bulk_Y = filtered_Bulk_T['label']   
    merged_data = CCL_ScRNA_meta.merge(Drug_AUC_cleaned, left_on='CVCL', right_index=True)
    merged_data['CVCL'] = pd.Categorical(merged_data['CVCL'], categories=filtered_Bulk_T.index, ordered=True)
    merged_data = merged_data.sort_values(by='CVCL')
    filtered_CCL_ScRNA = filtered_CCL_ScRNA.loc[merged_data['NAME'].values,:]
    label_CCL_ScRNA = merged_data['AUC'].values
    filtered_CCL_ScRNA = pd.DataFrame(filtered_CCL_ScRNA) 
    filtered_CCL_ScRNA['label_CCL_ScRNA'] = label_CCL_ScRNA
    CCL_ScRNA_X = filtered_CCL_ScRNA.drop(columns=['label_CCL_ScRNA'])  
    CCL_ScRNA_Y = filtered_CCL_ScRNA['label_CCL_ScRNA']
    Bulk_X_scaled = MinMaxScaler().fit_transform(Bulk_X.values)
    CCL_ScRNA_X_scaled = MinMaxScaler().fit_transform(CCL_ScRNA_X.values)
    X_single_cell_scaled = MinMaxScaler().fit_transform(filtered_SC_RNA.values)
    Bulk_X_scaled = torch.tensor(Bulk_X_scaled, dtype=torch.float32)
    Bulk_Y = torch.tensor(Bulk_Y.values, dtype=torch.long)
    CCL_ScRNA_X_scaled = torch.tensor(CCL_ScRNA_X_scaled, dtype=torch.float32)
    CCL_ScRNA_Y = torch.tensor(CCL_ScRNA_Y, dtype=torch.long)
    CCL_ScRNA_X_scaled, CCL_ScRNA_test, CCL_ScRNA_Y, CCL_ScRNA_test_Y = train_test_split(CCL_ScRNA_X_scaled,CCL_ScRNA_Y, random_state=42)
    X_single_cell_scaled = torch.tensor(X_single_cell_scaled, dtype=torch.float32)
    input_dim = Bulk_X_scaled.shape[1]   
    shared_encoder = SharedEncoder(input_dim, hidden_dim1)
    decoder_bulk = DomainDecoder(hidden_dim1, input_dim, domain_type='bulk')
    decoder_sc = DomainDecoder(hidden_dim1, input_dim, domain_type='sc')
    domain_classifier = AdversarialClassifier(hidden_dim1)
    model = MLP(hidden_dim1,hidden_dim1,hidden_dim2,hidden_dim3,output_dim,dropout) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(shared_encoder.parameters()) + 
        list(decoder_bulk.parameters()) + 
        list(decoder_sc.parameters()) + 
        list(domain_classifier.parameters()) + 
        list(model.parameters()),
        lr=l_r, 
        weight_decay=1e-5
    )
    for epoch in range(epochs):
        bulk = Bulk_X_scaled
        sc = X_single_cell_scaled
        optimizer.zero_grad()
        z_bulk = shared_encoder(bulk)
        z_sc = shared_encoder(sc)
        z_CCL_ScRNA = shared_encoder(CCL_ScRNA_X_scaled)
        recon_bulk = decoder_bulk(z_bulk)
        mean_sc, disp_sc, pi_sc = decoder_sc(z_sc)
        pred_auc = model(z_bulk)  
        pred_auc1 = model(z_CCL_ScRNA)  
        rev_z_bulk = GradientReversalFn.apply(z_bulk, 1.0)
        rev_z_sc = GradientReversalFn.apply(z_sc, 1.0)
        domain_pred = domain_classifier(torch.cat([rev_z_bulk, rev_z_sc]))
        loss_recon_bulk = nn.MSELoss()(recon_bulk, bulk)
        loss_recon_sc = ZINBLoss()(sc, mean_sc, disp_sc, pi_sc)
        loss_MLP_Bulk = nn.CrossEntropyLoss()(pred_auc, Bulk_Y)
        loss_MLP_CCL_ScRNA = nn.CrossEntropyLoss()(pred_auc1, CCL_ScRNA_Y)
        domain_labels = torch.cat([torch.zeros(z_bulk.size(0)), torch.ones(z_sc.size(0))])
        loss_adv = nn.BCELoss()(domain_pred.squeeze(), domain_labels)
        loss = (
            a1*loss_recon_bulk + 
            a2*loss_recon_sc + 
            b1 * loss_MLP_Bulk +  
            b2* loss_MLP_CCL_ScRNA +
            c * loss_adv    
        )
        z_CCL_ScRNA_test = shared_encoder(CCL_ScRNA_test)
        CCL_ScRNA_p = model(z_CCL_ScRNA_test)  
        CCL_ScRNA_p = torch.argmax(CCL_ScRNA_p, dim=1)
        loss.backward()
        optimizer.step()
        print(metrics.accuracy_score(CCL_ScRNA_p.detach().numpy(),CCL_ScRNA_test_Y),metrics.roc_auc_score(CCL_ScRNA_p.detach().numpy(),CCL_ScRNA_test_Y))
        print('Epoch [{}/{}], Loss: {:.4f}, loss_recon_bulk: {:.4f}, loss_recon_sc: {:.4f}, loss_MLP_Bulk: {:.4f} loss_MLP_CCL_ScRNA: {:.4f}, loss_adv: {:.4f}'.format(epoch+1, epochs, loss.item(), loss_recon_bulk.item(), loss_recon_sc.item(), loss_MLP_Bulk.item(),  loss_MLP_CCL_ScRNA.item(), loss_adv.item() ))
    return shared_encoder ,model,filtered_SC_RNA,sc











def log2TPM_1(SC_RNA):
    SC_RNA_T = SC_RNA.transpose()
    SC_RNA_T = SC_RNA_T.apply(lambda x: x/np.sum(x)*1e6, axis=1)
    SC_RNA_T = np.log2(SC_RNA_T + 1)
    SC_RNA = SC_RNA_T.transpose()
    return SC_RNA

def log2TPM_2(SC_RNA):
    SC_RNA_T = SC_RNA.transpose()
    SC_RNA_T = SC_RNA_T.apply(lambda x: x/np.sum(x)*1e6, axis=1)
    SC_RNA_T = np.log2(SC_RNA_T + 1)
    return SC_RNA_T










































