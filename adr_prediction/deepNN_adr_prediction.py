'''
Train deepNN models for adr prediction using 
a. Experimental transcriptomics 
b. Experimental proteomics
c. TransPro output proteomics.
note :
X = pertgx_builder.get_gx_only() for a,c with L1000 genes 
#X = pertgx_builder.get_ab_only() for b with available antibodies 
'''
import argparse
from adr_data_builder import ADRDataBuilder, PerturbedDGXDataBuilder, XYPreparer
import logging
import numpy as np
import pdb
from torch import device, cuda
from model_builder import DeepModel
import torch
import os
from numpy import random
import pickle

# check cuda
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--adr_file', help = 'the directory of file that have side effect data, *_PTs.csv',
                        default="data/Side_effect/FAERS_offsides_PTs.csv")
    parser.add_argument('--gx_file', help = 'the directory of file that have the experimental gene expression file, *DGX.csv, or proteomics file, pred or truth',
                        default="data/Side_effect/Prot_faers_experimental_x945.csv")
    parser.add_argument('--device',type=int, default=1)
    parser.add_argument('--exp_pros', type=int, default=0, help = 'whether the gx data is exp proteomics data')
    parser.add_argument('--seed',type=int,default=79)
    parser.add_argument('--cur_dataset',type=str,default='FAERS')
    parser.add_argument('--conf_level',type=float, default=0.5)
    args = parser.parse_args()

    shared_faers_low = ['BRD-A91699651',
                        'BRD-A97701745',
                        'BRD-K00259736',
                        'BRD-K38436528',
                        'BRD-K49328571',
                        'BRD-K63828191',
                        'BRD-K64052750',
                        'BRD-K79602928']

    shared_faers_high = shared_faers_low[:]+['BRD-K21680192']
    shared_sider_low = ['BRD-A91699651',
                        'BRD-A97701745',
                        'BRD-K38436528',
                        'BRD-K49328571',
                        'BRD-K63828191',
                        'BRD-K64052750',
                        'BRD-K79602928']
    shared_sider_high = shared_sider_low[:]+['BRD-K21680192']

    if args.cur_dataset =='FAERS':
        with open("data/Side_effect/train_trans_faers_low_drugs","rb") as fp:
            trans_train_faers_low = pickle.load(fp)
        with open("data/Side_effect/train_trans_faers_high_drugs","rb") as fp:
            trans_train_faers_high = pickle.load(fp)
        deep_model_layout = [400,400]
        deep_model_epochs = 250
        if args.conf_level==0.3:
            left_out_drugs = shared_faers_low
            trans_train_drugs = trans_train_faers_low
        else:
            left_out_drugs = shared_faers_high
            trans_train_drugs = trans_train_faers_high
    elif args.cur_dataset == 'SIDERS':
        with open("data/Side_effect/train_trans_sider_low_drugs","rb") as fp:
            trans_train_sider_low = pickle.load(fp)
        with open("data/Side_effect/train_trans_sider_high_drugs","rb") as fp:
            trans_train_sider_high = pickle.load(fp)
        args.adr_file = "data/Side_effect/SIDER_PTs.csv"
        deep_model_layout = [800,800]
        deep_model_epochs = 500
        if args.conf_level ==0.3:
            left_out_drugs = shared_sider_low
            trans_train_drugs = trans_train_sider_low
        else:
            left_out_drugs = shared_sider_high
            trans_train_drugs = trans_train_sider_high

    #print("Use GPU if it is deep model: %s" % torch.cuda.is_available())
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    adr_file = args.adr_file
    gx_file = args.gx_file
    pred_flag = args.pred_flag
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    adr_builder = ADRDataBuilder(file_name = adr_file)
    adr_drugs_list = adr_builder.get_drug_list()
    pertgx_builder = PerturbedDGXDataBuilder(gx_file_name=gx_file, pert_list = adr_drugs_list, pred_flag=pred_flag, cs_part = False)
    if args.exp_pros ==1: 
        X = pertgx_builder.get_ab_only()
    else: 
        X = pertgx_builder.get_gx_only() 
    X_pertid_list = pertgx_builder.get_pert_id_list()
    
    Y = adr_builder.prepare_adr_df_basedon_perts(X_pertid_list)

    data_preparer = XYPreparer(X, Y, X_pertid_list, args.seed)


    # set up logging system
    logging.basicConfig(filename='2023_logs/adr_prediction_'+args.cur_dataset + '_seed' +str(args.seed)+ '_' + str(args.conf_level)+'_' + gx_file.rsplit('/',1)[1].rsplit('.',1)[0].lower(), 
                    level=logging.DEBUG,
                    format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
    logger = logging.getLogger(name='ADR_Prediction')

    logger.debug("There are {0!r} drugs".format(len(set(X_pertid_list))))
    aucroc_ls=[]
    prauc_ls=[]
   
    test_rand = Y.loc[left_out_drugs]
    test_drug_idx =[ test_rand.reset_index().loc[test_rand.reset_index().pert_id==left_out_drugs[i]].index for i in range(len(left_out_drugs))]

    train_rand = Y.loc[trans_train_drugs]
    
    Y_truth_with_zero = test_rand.values
    Y_truth_train_with_zero = train_rand.values
    nozero_filter = (Y_truth_with_zero.sum(axis = 0) > 0) & (Y_truth_train_with_zero.sum(axis = 0) > 0)
    
    logger.debug("nozero cols is {0!r}".format(sum(nozero_filter)))
    
    bi_mask = train_rand.nunique()==2
    bi_mask_2 = test_rand.nunique()==2
    bad_cols = bi_mask[bi_mask==False].index.values
    bad_cols_2 = bi_mask_2[bi_mask_2==False].index.values
    bad_cols = np.concatenate([bad_cols,bad_cols_2])
    Y_train_pert = list(train_rand.drop(bad_cols,axis=1).index)
    Y_test_pert = list(test_rand.drop(bad_cols,axis=1).index)
    Y_train = train_rand.drop(bad_cols,axis=1).values
    Y_test =test_rand.drop(bad_cols,axis=1).values

    logger.debug("nozero and binary cols is {0!r}".format(Y_test.shape[1]))

   
    train_index = Y.reset_index().index[Y.reset_index()['pert_id'].isin(Y_train_pert)]
    test_index = Y.reset_index().index[Y.reset_index()['pert_id'].isin(Y_test_pert)]
    logger.debug("train data length {0!r}, test data length {1!r}".format(len(train_index), len(test_index)))
  
    cur_model = DeepModel(X_dim = X.shape[1], Y_dim = Y_test.shape[1], layout = deep_model_layout,random_seed=args.seed, device = device, logger = logger)
    Y_truth = Y_truth_with_zero[:, nozero_filter]
    if isinstance(cur_model, DeepModel):
        best_metric = -float('inf')
        aucroc_ls.append(float('inf'))
        prauc_ls.append(float('inf'))
        n_epochs = deep_model_epochs
    
        for i in range(n_epochs):
            loss = cur_model.fit(X.values[train_index,:], Y_train)
            aucroc, prauc = cur_model.score(X.values[test_index,:], Y_test, test_drug_idx)
            if aucroc > best_metric:
                best_metric = aucroc
                aucroc_ls[-1] = aucroc
                prauc_ls[-1] = prauc
                
        logger.debug("model prediction rocauc is {0:.4f}, prauc is {1:.4f}".format(aucroc_ls[-1], prauc_ls[-1]))
        
   
