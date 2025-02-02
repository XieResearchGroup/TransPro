from ast import parse
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
    parser.add_argument('--Ic_50_train_dir',
                        type = str,
                        default= 'data/Ic50/rand_0_train_ic50_wo_na.csv',
                        help= 'Ic50 training file')
    parser.add_argument('--Ic_50_dev_dir',
                        type = str,
                        default= 'data/Ic50/rand_0_dev_ic50_wo_na.csv',
                        help= 'Ic50 dev file')
    parser.add_argument('--Ic_50_test_dir',
                        type = str,
                        default= 'data/Ic50/rand_0123_test_ic50_wo_na.csv',
                        help= 'Ic50 test file')
    
    parser.add_argument('--drug_file_dir',
                         type = str, 
                         default='data/a_gdsc_drugs_smiles_pro.csv',
                         help = 'the drug file directory (# broad_id # smiles #)')
    parser.add_argument('--trans_basal_dir',
                         type = str, 
                         default='data/CCLE_x1305_978genes.csv',
                         help = 'basal transcriptome data (cell feature)')
    parser.add_argument('--pretrained_model_dir',
                        type = str,
                        default='data/trained_model/final_transPro_model.pt',
                        help = 'saved pretrained pretraining model')
                        # '/raid/home/yoyowu/PertPro/models_inventory/0422_get_pertTrans_w_transmitter_model.pt'
    parser.add_argument('--saved_model_path',
                        type = str,
                        default = None)
    parser.add_argument('--max_epochs',
                        type = int,
                        default=4,
                        help = 'Total number of epochs')    
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--wd',type=float, default=0.01)
    parser.add_argument('--include_trans', type=int, default=1,help='whether to include the pert trans data')                     
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--dop',type=float,default=0.1)
    parser.add_argument('--seed',type=int, default=343)
    parser.add_argument('--use_transmitter',type=int, default=1)
    parser.add_argument('--infer_mode',type=int, default=0,
                        help=' infer mode 0: infer mode is turned off, \
                        infer mode 1 : output the hidden representation, \
                        infer mode 2: output the final prediction')
    parser.add_argument('--task_spec', type = int, default=1)
    parser.add_argument('--job', type = str, default='perturbed_pros', help='which embedding to use, perturbed_pros or perturbed_trans')
    parser.add_argument('--freeze_pretrained_modules',type = int, default=0)
    parser.add_argument('--predicted_result_for_testset',type=str,default='/raid/home/yoyowu/CODE-AE/data/0608_Transpro_embeddiings_cle_gdsc_pred.csv')
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
    
    if args.infer_mode ==2 or args.infer_mode ==1:
        data_config.data_filter = None

    Ic_50_data = datareader.Ic_50_DataLoader(
            args.drug_file_dir, args.Ic_50_train_dir, args.Ic_50_dev_dir,
            args.Ic_50_test_dir,  data_config.data_filter, device,
            args.trans_basal_dir, batch_size = args.batch_size)
            
    Ic_50_data.setup()
    print('#Train: %d' % len(Ic_50_data.train_data))
    print('#Dev: %d' % len(Ic_50_data.dev_data))
    print('#Test: %d' % len(Ic_50_data.test_data))  


    ### prepare for models
    model_config = get_config('model')
    model = transPro_model.Ic50_task(
                                device, 
                                model_config,
                                args).double().to(device)
    
  
    if args.pretrained_model_dir:
        model.transPro.load_state_dict(torch.load(args.pretrained_model_dir))
        print("successfully loaded pretrained model from {}".format(args.pretrained_model_dir))
    model.config_optimizer( lr = args.lr)
    wandb.init(project="transPro_Ic50",config=args)
    wandb.watch(model, log="all")

    for epoch in range(args.max_epochs):
        print("Iteration %d:" % (epoch+1))
        for step, (features,labels,_) in enumerate(Ic_50_data.train_dataloader()):
            model.train_step(
                features['drug'].to(device),
                features['cell_id'],
                labels,job =args.job,epoch=epoch,
                freeze_pretrained_modules=args.freeze_pretrained_modules
            )
        model.train_epoch_end(epoch)

        for step, (features,labels,_) in enumerate(Ic_50_data.val_dataloader()):
            model.val_test_step(
                    features['drug'].to(device),
                    features['cell_id'],
                    labels,job = args.job,epoch=epoch
                )
        model.validation_test_epoch_end(epoch=epoch, 
                                    validation_test_flag ='Perturbed_Pros_Validation',
                                    metrics_summary=metrics_summary )

        for step, (features,labels,_) in enumerate(Ic_50_data.test_dataloader()):
            model.val_test_step(
                    features['drug'].to(device),
                    features['cell_id'],
                    labels,job =args.job,epoch=epoch
                )
        model.validation_test_epoch_end(epoch=epoch, 
                                validation_test_flag ='Perturbed_Pros_Test',
                                metrics_summary=metrics_summary )
    
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

