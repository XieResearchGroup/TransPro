#!/usr/bin/env bash
#split=stadScaf  #cell_split_4 cell_split_2 cell_cluster_1s9 stadScaf rand_split_119 cell_split_1 cell_split_2 
#for split in stadScaf
#do
#for seed in 27 72 
#do 
exp_id=0422_get_pertTrans_w_transmitter_model
split=cell_split_1
python ../perturbed_proteomics/transPro.py \
--exp_id ${exp_id} --dop 0.2 --seed 343 \
--device 0  --warmup_epochs 600 --max_epochs 1500 --lr_low 0.0001 --lr_high 0.0002 --include_trans 1 --use_transmitter 1 \
--pert_trans_train_dir "../perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_full.csv" \
--pert_trans_dev_dir "../perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_dev.csv" \
--pert_trans_test_dir "../perturbed_proteomics/data/pert_pro_plus_pert_trans/adjusted_l1000_pert_trans_test.csv" \
--pert_pros_train_dir "/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/${split}_512ab_noImpu_pert_pros_dev.csv" \
--pert_pros_dev_dir "/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/${split}_512ab_pert_pros_dev.csv" \
--pert_pros_test_dir "/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/${split}_512ab_pert_pros_test.csv" \
--drug_file_dir "../perturbed_proteomics/data/drugs_smiles_pro.csv" \
--trans_basal_dir "../perturbed_proteomics/data/Combat_batch_removal/fixed_adjusted_ccle_tcga_basal_trans.csv" \
--saved_model_path "/raid/home/yoyowu/PertPro/models_inventory/${exp_id}.pt" \
>../Apr_logs/${exp_id}_mixlr_seed_${seed}.log 2>&1 &
#done
#done
#--pretrained_model_dir "/raid/home/yoyowu/PertPro/models_inventory/0206_ginAttn_trans_2k_4e-5_dop0.3.pt" \
#--pretrained_modules "/raid/home/yoyowu/PertPro/models_inventory/0219_cell_split_1_pretrain_no_transmitter_500_pt" \
#--pretrained_modules "/raid/home/yoyowu/PertPro/models_inventory/0225_pretrain_wo_transmitter_Imputed_cell_split_1_500_pt"
#--pretrained_modules "/raid/home/yoyowu/PertPro/models_inventory/0226_pretrain_wo_transmitter_cell_split_2_500_pt" \
#--saved_model_path "/raid/home/yoyowu/PertPro/models_inventory/${exp_id}.pt" \
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/pert_pro_plus_pert_trans/total_512ab_pert_pros.csv