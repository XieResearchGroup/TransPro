import os
import sys
from xmlrpc.client import boolean
import torch
import argparse
import pandas as pd 
import datareader
import numpy as np 
import wandb
from transPro_config import get_config
import transPro_model
from collections import defaultdict
from torch import save 
import random


metrics_summary = defaultdict(
    pearson_list_dev = [],
    pearson_list_test = [],
    spearman_list_dev = [],
    spearman_list_test = [],
    rmse_list_dev = [],
    rmse_list_test = [])
# check cuda

def setup_dataloader(dataloader):
    dataloader.setup()
    print('#Train: %d' % len(dataloader.train_data))
    print('#Dev: %d' % len(dataloader.dev_data))
    print('#Test: %d' % len(dataloader.test_data))    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'transPro')
    parser.add_argument('--exp_id', type=str, default='test')
    parser.add_argument('--pert_trans_train_dir',
                         type = str, 
                         default='perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_full.csv',
                         help = 'perturbed transcriptome data for training')
    parser.add_argument('--pert_trans_dev_dir',
                         type = str, 
                         default='perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_dev.csv',
                         help = 'perturbed transcriptome data for dev')
    parser.add_argument('--pert_trans_test_dir',
                         type = str, 
                         default='perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_test.csv',
                         help = 'perturbed transcriptome data for test')

    parser.add_argument('--pert_pros_train_dir',
                         type = str, 
                         default='/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/cell_split_1_512ab_noImpu_pert_pros_train.csv',
                         help = 'perturbed proteomics data for training')
    parser.add_argument('--pert_pros_dev_dir',
                         type = str, 
                         default='/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/cell_split_1_512ab_pert_pros_dev.csv',
                         help = 'perturbed proteomics data for dev')
    parser.add_argument('--pert_pros_test_dir',
                         type = str, 
                         default='/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/cell_split_1_512ab_pert_pros_test.csv',
                         help = 'perturbed proteomics data for test, can be used to infer on other data ')
                         #/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/FAERS_PTs_allCells_0.3conf_3.33um_CellLineDGX_x2142.csv
                         #/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/FAERS_PTs_allCells_0.5conf_3.33um_CellLineDGX_x2192.csv
    parser.add_argument('--drug_file_dir',
                         type = str, 
                         default='perturbed_proteomics/data/drugs_smiles_pro.csv',
                         help = 'the drug file directory (# broad_id # smiles #)')
    parser.add_argument('--trans_basal_dir',
                         type = str, 
                         default='perturbed_proteomics/data/Combat_batch_removal/fixed_adjusted_ccle_tcga_basal_trans.csv',
                         help = 'basal transcriptome data (cell feature)')
    parser.add_argument('--pretrained_model_dir',
                        type = str,
                        default= None,
                        help = 'saved pretrained pretraining model')
                        # '/raid/home/yoyowu/PertPro/models_inventory/0422_get_pertTrans_w_transmitter_model.pt'
    parser.add_argument('--saved_model_path',
                        type = str,
                        default = None)
    parser.add_argument('--warmup_epochs',type=int, default=2,help='the epochs for altanative training with pert trans data, default:600')
    parser.add_argument('--max_epochs',
                        type = int,
                        default=4,
                        help = 'Total number of epochs')    
    parser.add_argument('--lr_low',type=float,default=0.0001)
    parser.add_argument('--lr_high',type=float,default=0.0002)
    parser.add_argument('--wd',type=float, default=0.01)
    parser.add_argument('--include_trans', type=int, default=1,help='whether to include the pert trans data')                     
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--dop',type=float,default=0.1)
    parser.add_argument('--seed',type=int, default=343)
    parser.add_argument('--use_transmitter',type=int, default=1)
    parser.add_argument('--infer_mode',type=int, default=0,help=' infer mode 0: infer mode is turned off, infer mode 1 : output the hidden representation, infer mode 2: output the final prediction')
    parser.add_argument('--freeze_pretrained_modules',type = int, default=0)
    parser.add_argument('--predicted_result_for_testset',type=str,default='/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423Pros_preds_512ab_FAERS_PTs_allCells_0.5conf_3.33um_w_pertTrans.csv')
 # /raid/home/yoyowu/PertPro/chemblFiltered_and_supervise_pretrained_model_with_contextPred.pth
    args = parser.parse_args()
    
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print("Use GPU: %s" % args.device)
    ### prepare for the two dataloaders

    

    data_config = get_config('data')
    
    if args.infer_mode ==2:
        data_config.data_filter = None

    pert_pros_dataloader = datareader.PerturbedDataLoader(
        args.drug_file_dir, args.pert_pros_train_dir, args.pert_pros_dev_dir,
        args.pert_pros_test_dir, data_config.data_filter, device,
        args.trans_basal_dir, batch_size = 64
    )
    setup_dataloader(pert_pros_dataloader)

    if args.include_trans ==1:
        pert_trans_dataloader = datareader.PerturbedDataLoader(
            args.drug_file_dir, args.pert_trans_train_dir, args.pert_trans_dev_dir,
            args.pert_trans_test_dir,  data_config.data_filter, device,
            args.trans_basal_dir, batch_size = 64)

        setup_dataloader(pert_trans_dataloader)

    ### prepare for models
    model_config = get_config('model')
    model = transPro_model.TransProModel(
                                device, 
                                model_config,
                                args).double().to(device)
    
  
    if args.pretrained_model_dir:
        model.load_state_dict(torch.load(args.pretrained_model_dir))
        print("successfully loaded pretrained model from {}".format(args.pretrained_model_dir))
    
    wandb.init(project="trans_pros_pretraining",config=args)
    wandb.watch(model, log="all")

    if args.infer_mode==1:
        for step, (features, labels, _) in enumerate(pert_pros_dataloader.test_dataloader()):
            model.perturbed_pros_val_test_step(
                features['drug'].to(device),
                features['cell_id'],
                labels
            )
        lb_np, predict_np = np.concatenate(model.label_ls), np.concatenate(model.prediction_ls)
        sorted_test_input = pd.read_csv(args.pert_pros_test_dir).sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])
        #genes_cols = pd.read_csv(args.pert_pros_dev_dir).columns[5:]
        assert sorted_test_input.shape[0] == predict_np.shape[0]
        predict_df = pd.DataFrame(predict_np, index = sorted_test_input.index)
        result_df  = pd.concat([sorted_test_input.iloc[:, :5], predict_df], axis = 1)
        
        print("=====================================write out data=====================================")
        result_df.loc[[x for x in range(len(result_df))],:].to_csv(args.predicted_result_for_testset, index = False)
    
    elif args.infer_mode==2:
        
        for step, features in enumerate(pert_pros_dataloader.test_dataloader()):
            model.perturbed_pros_val_test_step(
                features['drug'].to(device),
                features['cell_id'] ,labels=None)
        predict_np = np.concatenate(model.prediction_ls)
        sorted_test_input = pd.read_csv(args.pert_pros_test_dir).sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])
        genes_cols = pd.read_csv(args.pert_pros_dev_dir).columns[5:]
        assert sorted_test_input.shape[0] == predict_np.shape[0]
        predict_df = pd.DataFrame(predict_np, index = sorted_test_input.index,columns=genes_cols)
        result_df  = pd.concat([sorted_test_input.iloc[:, :5], predict_df], axis = 1)

        print("=====================================write out data=====================================")
        result_df.loc[[x for x in range(len(result_df))],:].to_csv(args.predicted_result_for_testset, index = False)
    
    else:
        # start training... 
        ## set lower learning rate
        model.config_optimizer(lr = args.lr_low)
        for epoch in range( args.warmup_epochs):
            print("Iteration %d:" % (epoch+1))
            if args.include_trans ==1:
                print('Including trans training...')
                if epoch % 5==0:
                    print('Perturbed Train Val Trans....')
                    for step, (features, labels, _) in enumerate(pert_trans_dataloader.train_dataloader()):
                        model.perturbed_trans_train_step(
                            features['drug'].to(device),
                            features['cell_id'],
                            labels,
                            epoch)
                    model.train_epoch_end(epoch)

                    for step, (features, labels, _) in enumerate(pert_trans_dataloader.val_dataloader()):
                        model.perturbed_trans_val_test_step(
                            features['drug'].to(device),
                            features['cell_id'],
                            labels,
                            epoch,
                        )
                    model.validation_test_epoch_end(epoch = epoch,
                                                            validation_test_flag = 'Perturbed_Trans_Validation',metrics_summary=metrics_summary)

                print('Perturbed Train Val Pros....')
                for step, (features, labels, _) in enumerate(pert_pros_dataloader.train_dataloader()):
                    model.perturbed_pros_train_step(
                        features['drug'].to(device),
                        features['cell_id'],
                        labels,
                        epoch,freeze_pretrained_modules=args.freeze_pretrained_modules
                    )
                model.train_epoch_end(epoch)

                for step, (features, labels, _) in enumerate(pert_pros_dataloader.val_dataloader()):
                    model.perturbed_pros_val_test_step(
                        features['drug'].to(device),
                        features['cell_id'],
                        labels,
                        epoch,
                    )
                model.validation_test_epoch_end(epoch = epoch,
                                                        validation_test_flag = 'Perturbed_Pros_Validation',metrics_summary=metrics_summary)
                                                        #model_persistence_dir = args.saved_model_path)
            if args.include_trans ==1:
                if epoch % 5==0:
                    print('Perturbed Test Trans....')
                    for step, (features, labels, _) in enumerate(pert_trans_dataloader.test_dataloader()):
                        model.perturbed_trans_val_test_step(
                            features['drug'].to(device),
                            features['cell_id'],
                            labels,
                            epoch)
                    model.validation_test_epoch_end(epoch = epoch,
                                                            validation_test_flag = 'Perturbed_Trans_Test',metrics_summary=metrics_summary)

                print('Perturbed Test Pros....')
                for step, (features, labels, _) in enumerate(pert_pros_dataloader.test_dataloader()):
                    model.perturbed_pros_val_test_step(
                        features['drug'].to(device),
                        features['cell_id'],
                        labels,
                        epoch)
                model.validation_test_epoch_end(epoch = epoch,
                                                        validation_test_flag = 'Perturbed_Pros_Test',metrics_summary=metrics_summary)
        model.config_optimizer(lr = args.lr_high)
        for epoch in range( args.warmup_epochs, args.max_epochs):
            print("Iteration %d:" % (epoch+1))    
            print('Perturbed Train Val Pros....')
            for step, (features, labels, _) in enumerate(pert_pros_dataloader.train_dataloader()):
                model.perturbed_pros_train_step(
                    features['drug'].to(device),
                    features['cell_id'],
                    labels,
                    epoch,freeze_pretrained_modules=args.freeze_pretrained_modules
                )
            model.train_epoch_end(epoch)

            for step, (features, labels, _) in enumerate(pert_pros_dataloader.val_dataloader()):
                model.perturbed_pros_val_test_step(
                    features['drug'].to(device),
                    features['cell_id'],
                    labels,
                    epoch,
                )
            model.validation_test_epoch_end(epoch = epoch,
                                                    validation_test_flag = 'Perturbed_Pros_Validation',metrics_summary=metrics_summary)
                                                    #model_persistence_dir = args.saved_model_path)                                        
            print('Perturbed Test Pros....')
            for step, (features, labels, _) in enumerate(pert_pros_dataloader.test_dataloader()):
                model.perturbed_pros_val_test_step(
                    features['drug'].to(device),
                    features['cell_id'],
                    labels,
                    epoch)
            model.validation_test_epoch_end(epoch = epoch,
                                                    validation_test_flag = 'Perturbed_Pros_Test',metrics_summary=metrics_summary)
        if args.saved_model_path:
            save(model.state_dict(),args.saved_model_path)
            print("the trained model is successfully saved at {}".format(args.saved_model_path))

        best_dev_epoch = np.argmax(metrics_summary['pearson_list_dev'])


        print("Epoch %d got best Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_list_dev'][best_dev_epoch]))
        print("Epoch %d got Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_list_dev'][best_dev_epoch]))
        print("Epoch %d got RMSE on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_dev'][best_dev_epoch]))



        print("Epoch %d got Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_list_test'][best_dev_epoch]))
        print("Epoch %d got Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_list_test'][best_dev_epoch]))
        print("Epoch %d got RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_test'][best_dev_epoch]))


        best_test_epoch = np.argmax(metrics_summary['pearson_list_test'])


        print("Epoch %d got best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, metrics_summary['pearson_list_test'][best_test_epoch]))
        print("Epoch %d got  Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, metrics_summary['spearman_list_test'][best_test_epoch]))
        print("Epoch %d got  RMSE on test set: %.4f" % (best_test_epoch + 1, metrics_summary['rmse_list_test'][best_test_epoch]))

