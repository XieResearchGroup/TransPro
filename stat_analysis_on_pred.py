import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
from collections import defaultdict
from scipy.stats import hypergeom
import csv
pd.options.mode.chained_assignment = None


def save_csv(data,file_path):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)    
        for k,v in data.items():
            writer.writerow([k] + v)

def ks_test():
    print('The ks test result of cell {} on {} drug rsps features'.format(current_cell,len(nonull_drugs)))
    # sig_cluster=[]
    # sig_pvalue = []
    for drug_cluster in range(clustered_res.cluster.nunique()):
    
        k_cluster = clustered_res.loc[clustered_res.cluster==drug_cluster].pert_id
        
        assert(len(k_cluster)>0)
        exp_data = current_ic50[list(k_cluster.values)].values.flatten()
        p_val = ks_2samp(exp_data,current_ic50.values)[1] 
        # print('cluster {} contains {} drugs, the p value is'.format(drug_cluster,len(k_cluster)))
        # print(p_val)
        if p_val <0.05:
            print('found cluster{} has significant p value {}'.format(drug_cluster,p_val))
            ks_res[current_cell].append(len(nonull_drugs))
            ks_res[current_cell].append(drug_cluster)
            ks_res[current_cell].append(len(k_cluster))
            ks_res[current_cell].append(p_val)   

def hypergeo_test():
   
    
    #clustered_res.loc[clustered_res.cluster==0].pert_id.intersection(topk_drugs)
    print('The hypergeometric test result of cell {} on {} drug rsps features'.format(current_cell,len(nonull_drugs)))
    
   
    for drug_cluster in range(clustered_res.cluster.nunique()):
        k_cluster = clustered_res.loc[clustered_res.cluster==drug_cluster].pert_id
        k = len(k_cluster)
        #print('The current k== the number of drugs is {}'.format(k))
        bottomk_drugs = current_ic50.sort_values(ascending=False)[:k].index
        topk_drugs = current_ic50.sort_values(ascending=True)[:k].index
        x_top = np.intersect1d(k_cluster,topk_drugs).shape[0]
        x_bottom = np.intersect1d(k_cluster,bottomk_drugs).shape[0]
        M = len(nonull_drugs)
        n = k
        N = len(k_cluster)
        topk_pval= hypergeom.sf(x_top-1, M, n, N)
        bottomk_pval = hypergeom.sf(x_bottom-1, M, n, N)
        if topk_pval <0.05:
            print('found significant p ')
            print('cluster {}  top k has {}/{} success , and the pval is {}'.format(drug_cluster,x_top,N,topk_pval))
            #print('The current k== the number of drugs is {}'.format(k))
            hyper_top_res[current_cell].append(len(nonull_drugs))
            hyper_top_res[current_cell].append(drug_cluster)
            hyper_top_res[current_cell].append(N)
            hyper_top_res[current_cell].append(topk_pval)   
        elif bottomk_pval < 0.05:
            print('found significant p ')
            print('cluster {}  bottom k has {}/{} success , and the pval is {}'.format(drug_cluster,x_bottom,N,bottomk_pval))
            #print('The current k== the number of drugs is {}'.format(k))
            hyper_bot_res[current_cell].append(len(nonull_drugs))
            hyper_bot_res[current_cell].append(drug_cluster)
            hyper_bot_res[current_cell].append(N)
            hyper_bot_res[current_cell].append(bottomk_pval)    
        # print('cluster {}  top k has {}/{} success , and the pval is {}'.format(drug_cluster,x_top,N,topk_pval))
        # print('cluster {}  bottom k has {}/{} success , and the pval is {}'.format(drug_cluster,x_bottom,N,bottomk_pval))


# load the prediction w CCLE/GDSC available data 680 cells , 370 drugs
transPro_pred_result = pd.read_csv('/raid/home/yoyowu/PertPro/perturbed_proteomics/data/0701_trans_predictions.csv')
#transPro_train_data = pd.read_csv('/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/total_512ab_pert_pros.csv')
#cells_used_train = set(transPro_train_data.cell_id)& set(transPro_pred_result.cell_id)
all_cells = transPro_pred_result.cell_id.tolist()
res_df_to_cluster = transPro_pred_result.loc[transPro_pred_result.cell_id.isin(all_cells)]
#cell_to_cluster_lst = list(all_cells)

ic_50 = pd.read_csv('/raid/home/yoyowu/CODE-AE/data/0609_ic50_ccle_gdsc_c680_d370.csv')
#auc = pd.read_csv('/raid/home/yoyowu/CODE-AE/data/0609_auc_ccle_gdsc_c680_d370.csv')

cell_name_map = pd.read_csv('/raid/home/yoyowu/CODE-AE/data/0609_cell_name_mapping.csv')
ic_50.COSMIC_ID = ic_50.COSMIC_ID.map(cell_name_map.set_index('DepMap_ID')['stripped_cell_line_name'].get)
ic_50 = ic_50.set_index('COSMIC_ID')
ic_50= ic_50.div(ic_50.sum(axis=1), axis=0)

ks_res= defaultdict(list)
hyper_top_res = defaultdict(list)
hyper_bot_res = defaultdict(list)

# choose which cell o focus on 
#df = res_df_to_cluster.loc[res_df_to_cluster.cell_id==cell_to_cluster_lst[0]]

#for i in range(len(ic_50)):
for i in range(len(ic_50)):
    current_cell = ic_50.index[i]
    #current_cell = 'SKMEL28'
    print('--------------------------------------------iteration : {}--------------------------------'.format(i))
    print('current cell is {}'.format(current_cell))
    df = res_df_to_cluster.loc[res_df_to_cluster.cell_id==current_cell]
    a = ic_50.loc[current_cell].notnull()
    nonull_drugs = a[a].index.tolist()
    current_ic50 = ic_50.loc[current_cell].dropna()
    print(' the current cell has {} drug rsps features'.format(len(nonull_drugs)))
    assert (df[df.pert_id.isin(nonull_drugs)].pert_id.nunique() == len(nonull_drugs))
    df_nonull_drugs = df[df.pert_id.isin(nonull_drugs)]

    # using global k as  7
    data = df_nonull_drugs.iloc[:,2:]
    kmeans=KMeans(n_clusters=7, random_state=27)
    kmeans.fit(data)
    df_nonull_drugs['cluster'] =  kmeans.labels_
    clustered_res = df_nonull_drugs.sort_values('cluster')[['pert_id','cluster']]

    hypergeo_test()
    ks_test()

save_csv(ks_res,'/raid/home/yoyowu/PertPro/perturbed_proteomics/data/0629_stat_pred_res/0701_trans_ks_res.csv')
print('-------------successfully saved ks test result!------------')
save_csv(hyper_top_res,'/raid/home/yoyowu/PertPro/perturbed_proteomics/data/0629_stat_pred_res/0701_trans_hyper_top_res.csv')
print('-------------successfully saved hyper top test result!------------')
save_csv(hyper_bot_res,'/raid/home/yoyowu/PertPro/perturbed_proteomics/data/0629_stat_pred_res/0701_trans_hyper_bot_res.csv')
print('-------------successfully saved hyper bot test result!------------')