import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings


def calculate_cutoff(auc_values):
    valid_auc = auc_values[~np.isnan(auc_values)]  
    if len(valid_auc) < 2:
        return np.nan
    
    sorted_auc = np.sort(valid_auc)[::-1]  
    x = np.arange(len(sorted_auc)).reshape(-1, 1)
    pearson_corr, _ = pearsonr(sorted_auc, x.ravel().astype(float))
    if abs(pearson_corr) > 0.9:
        return np.median(sorted_auc)
    else:
        max_auc, min_auc = sorted_auc[0], sorted_auc[-1]
        n = len(sorted_auc)
        def distance(x_point, y_point):
            a = min_auc - max_auc
            b = n - 0
            c = max_auc * 0 - min_auc * n
            return abs(a*x_point + b*y_point + c) / np.sqrt(a**2 + b**2)
        
        distances = [distance(i, sorted_auc[i]) for i in range(n)]
        max_dist_idx = np.argmax(distances)
        return sorted_auc[max_dist_idx]


def Binary_Label_Classification(AUC):
    warnings.filterwarnings("ignore")
    df_auc = AUC
    binary_labels = pd.DataFrame(index=df_auc.index)  
    for drug in df_auc.columns:
        auc_series = df_auc[drug]
        cutoff = calculate_cutoff(auc_series.values)
        if cutoff >= 0.95:
            cutoff = 0.95
        if np.isnan(cutoff):
            binary_labels[drug] = np.nan
        else:
            nan_mask = auc_series.isna()
            response = (auc_series >= cutoff).astype(float) 
            response = response.replace({1.0: 1, 0.0: 0}) 
            response[nan_mask] = np.nan
            binary_labels[drug] = response
    MIN_MINORITY = 50       
    MINORITY_RATIO = 0.2     
    class_counts = binary_labels.apply(lambda col: col.value_counts(dropna=True))
    minority_counts = class_counts.min(axis=0)          
    total_counts = class_counts.sum(axis=0)             
    minority_ratio = minority_counts / total_counts   
    print(pd.DataFrame({
        "0_count": class_counts.loc[0],
        "1_count": class_counts.loc[1],
        "minority_ratio": minority_ratio
    }).head())
    unbalanced_drugs = (
        (minority_counts < MIN_MINORITY) |  
        (minority_ratio < MINORITY_RATIO)   
    )
    drugs_to_drop = minority_counts[unbalanced_drugs].index.tolist()
    print(f"{len(drugs_to_drop)}")
    print( drugs_to_drop)
    balanced_labels = binary_labels.drop(columns=drugs_to_drop)
    balanced_auc = df_auc.drop(columns=drugs_to_drop)  
    print(, balanced_labels.shape)
    AUC = balanced_labels
    return AUC














































