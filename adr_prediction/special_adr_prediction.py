'''
Train different models on the adr prediction dataset, this script is used for multi-label strategies
SVM, RF, Logistic
'''
import argparse
from adr_data_builder import ADRDataBuilder, PerturbedDGXDataBuilder, XYPreparer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier,LogisticRegression, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import numpy as np
import pdb
from torch import device, cuda
from model_builder import DeepModel
import torch
import os
import wandb
from numpy import random

# check cuda
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--adr_file', help = 'the directory of file that have side effect data, *_PTs.csv',default="/raid/home/yoyowu/MultiDCP/MultiDCP_data/side_effect/FAERS_offsides_PTs.csv")
    parser.add_argument('--gx_file', help = 'the directory of file that have the gene expression file, pred or truth, *DGX.csv',default="/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/FAERS_PTs_allCells_0.3conf_3.33um_CellLineDGX_x2142.csv")
    parser.add_argument('--cs_file', help = 'the directory of file that have MACC chemical strucutre information, MACC_bitmatrix')
    parser.add_argument('--device',type=int, default=2)
    parser.add_argument('--pred_data', dest = 'pred_flag', action='store_true', default=False,
                    help = 'whether the gx data is perdicted data')
    parser.add_argument('--fold', help = 'the fold to be test in this script', type=int,default=8)
    parser.add_argument('--seed',type=int,default=79)
    parser.add_argument('--split_method',type=str, default="manual_rand")
    parser.add_argument('--cur_dataset',type=str,default='FAERS')
    parser.add_argument('--conf_level',type=float, default=0.3)
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
        deep_model_layout = [400,400]
        deep_model_epochs = 500
        if args.conf_level==0.3:
            left_out_drugs = shared_faers_low
        else:
            left_out_drugs = shared_faers_high
    elif args.cur_dataset == 'SIDERS':
        args.adr_file = "/raid/home/yoyowu/MultiDCP/MultiDCP_data/side_effect/SIDER_PTs.csv"
        deep_model_layout = [800,800]
        deep_model_epochs = 500
        if args.conf_level ==0.3:
            left_out_drugs = shared_sider_low
        else:
            left_out_drugs = shared_sider_high

    #print("Use GPU if it is deep model: %s" % torch.cuda.is_available())
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    adr_file = args.adr_file
    gx_file = args.gx_file
    pred_flag = args.pred_flag
    cs_file = args.cs_file
    fold = args.fold if args.fold < 5 else None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    adr_builder = ADRDataBuilder(file_name = adr_file)
    adr_drugs_list = adr_builder.get_drug_list()
    pertgx_builder = PerturbedDGXDataBuilder(gx_file_name=gx_file, drug_cs_dir=cs_file, pert_list = adr_drugs_list, pred_flag=pred_flag, cs_part = False)

    multi_label_models = [RandomForestClassifier, ExtraTreesClassifier, RidgeClassifierCV, MLPClassifier, KNeighborsClassifier]
    
    X = pertgx_builder.get_gx_only()
    #X = pertgx_builder.get_ab_only()
    
    X_pertid_list = pertgx_builder.get_pert_id_list()
    
    Y = adr_builder.prepare_adr_df_basedon_perts(X_pertid_list)

    data_preparer = XYPreparer(X, Y, X_pertid_list, args.seed)

    cur_model_name = 'deep'

    # set up logging system
    logging.basicConfig(filename='/raid/home/yoyowu/PertPro/adr_prediction/logs/deepNN/manually_leave_one_drug_out/'+args.cur_dataset +'_'+ args.split_method +'_'+ cur_model_name.lower() + '_seed' +str(args.seed)+ '_' + str(args.conf_level)+'_' + gx_file.rsplit('/',1)[1].rsplit('.',1)[0].lower(), 
                    level=logging.DEBUG,
                    format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
    logger = logging.getLogger(name='ADR_Prediction')

    logger.debug("There are {0!r} drugs".format(len(set(X_pertid_list))))
    accuracy_micro_ls = []
    prauc_micro_ls = []
    accuracy_macro_ls = []
    prauc_macro_ls = []
    logger.debug("start 5-fold CV")
    #for split_num, (train_index, test_index) in enumerate(data_preparer.leave_new_drug_out_split()):
        #if fold and split_num != fold:
        
    # if args.split_method == "scaffold_split":
    #     train_index,dev_index,test_index = data_preparer.scaffold_split(X,data_preparer.smiles_all)
    #     logger.debug("using scaffold split now ")
    # elif args.split_method == "rand_scaffold_split":
    #     train_index,dev_index,test_index = data_preparer.random_scaffold_split(X,data_preparer.smiles_all,seed=args.seed)
    #     logger.debug("using random scaffold split now ")
    #logger.debug("the {0!r} split".format(split_num)) 
    split_num = 0
    


    test_rand = Y.loc[left_out_drugs]
    
    train_rand = Y[~Y.index.isin(left_out_drugs)]
    
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
    #cur_model = RandomForestClassifier(n_estimators = 100, verbose = 2, n_jobs = 20,random_state=args.seed)
    #cur_model = MLPClassifier(hidden_layer_sizes=200, verbose = 1,max_iter=500)
    #cur_model = ExtraTreesClassifier(n_estimators = 100, verbose = 2, n_jobs = 40,random_state=args.seed)
    #cur_model = KNeighborsClassifier()
    cur_model = DeepModel(X_dim = X.shape[1], Y_dim = Y_test.shape[1], layout = deep_model_layout,random_seed=args.seed, device = device, split_num=split_num, logger = logger)
    Y_truth = Y_truth_with_zero[:, nozero_filter]
    if isinstance(cur_model, DeepModel):
        best_metric = -float('inf')
        accuracy_micro_ls.append(float('inf'))
        accuracy_macro_ls.append(float('inf'))
        prauc_micro_ls.append(float('inf'))
        prauc_macro_ls.append(float('inf'))
        n_epochs = deep_model_epochs
    
        for i in range(n_epochs):
            # loss = cur_model.fit(X.values[train_index,:], Y.values[train_index,:][:, nozero_filter])
            # aucroc_micro, aucroc_macro, micro_prauc, macro_prauc = cur_model.score(X.values[test_index,:], Y.values[test_index,:][:, nozero_filter])
            loss = cur_model.fit(X.values[train_index,:], Y_train)
            aucroc_micro, aucroc_macro, micro_prauc, macro_prauc = cur_model.score(X.values[test_index,:], Y_test)
            #print('sucessfully predicting!')
            if aucroc_macro > best_metric:
                best_metric = aucroc_macro
                accuracy_micro_ls[-1] = aucroc_micro
                prauc_micro_ls[-1] = micro_prauc
                accuracy_macro_ls[-1] = aucroc_macro
                prauc_macro_ls[-1] = macro_prauc
        logger.debug("model prediction micro rocauc is {0:.4f}, macro rocauc is {1:.4f}".format(accuracy_micro_ls[-1], accuracy_macro_ls[-1]))
        logger.debug("model prediction micro prauc is {0:.4f}, macro prauc is {1:.4f}".format(prauc_micro_ls[-1], prauc_macro_ls[-1]))
    
    else:
        cur_model.fit(X.values[train_index,:], Y.values[train_index,:][:, nozero_filter])
        #Y_pred = cur_model.predict_proba(X.values[test_index,:])
        Y_pred = cur_model.predict(X.values[test_index,:])
        logger.debug("Predict successfully")
        if isinstance(Y_pred, list):
            Y_pred = np.transpose([pred[:, 1] for pred in Y_pred])
        # Y_pred = np.transpose([pred[:, 1] for pred in Y_pred])
        aucroc_micro = roc_auc_score(Y_truth.reshape(-1), Y_pred.reshape(-1))
        prauc_micro = average_precision_score(Y_truth.reshape(-1), Y_pred.reshape(-1))
        accuracy_micro_ls.append(aucroc_micro)
        prauc_micro_ls.append(prauc_micro)
        aucroc_macro = roc_auc_score(Y_truth, Y_pred, average = 'weighted')
        prauc_macro = average_precision_score(Y_truth, Y_pred, average = 'weighted')
        accuracy_macro_ls.append(aucroc_macro)
        prauc_macro_ls.append(prauc_macro)
        logger.debug("model prediction micro rocauc is {0:.4f}, macro rocauc is {1:.4f}".format(aucroc_micro, aucroc_macro))
        logger.debug("model prediction micro prauc is {0:.4f}, macro prauc is {1:.4f}".format(prauc_micro, prauc_macro))
    avg_aucroc_micro = np.mean(accuracy_micro_ls)
    avg_prauc_micro = np.mean(prauc_micro_ls)
    avg_aucroc_macro = np.mean(accuracy_macro_ls)
    avg_prauc_macro = np.mean(prauc_macro_ls)

    logger.debug("model prediction mean micro rocauc is {0:.4f}, mean macro rocauc is {1:.4f}".format(avg_aucroc_micro, avg_aucroc_macro))
    logger.debug("model prediction mean micro prauc is {0:.4f}, mean macro prauc is {1:.4f}".format(avg_prauc_micro, avg_prauc_macro))
